import numpy as np
import pandas as pd
import os
import PIL
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split


def split(df, random_state):
    column = "label"
    df_train, df_dummy = train_test_split(df, train_size=0.6, shuffle=True, random_state=random_state, stratify=df[column])
    df_valid, df_test = train_test_split(df_dummy, train_size=0.5, shuffle=True, random_state=random_state, stratify=df_dummy[column])
    print(f"Final sizes - train: {len(df_train)} valid: {len(df_valid)} test: {len(df_test)}")
    return df_train, df_valid, df_test

def create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"{directory_name} created.")
    else:
        print(f"{directory_name} aready exist.")

def resize(input_image_path, output_image_path, width, height):
    with Image.open(input_image_path) as img:
        if img.mode == "P":
            img = img.convert("RGB")
        resized_img = img.resize((width, height))
        resized_img.save(output_image_path)

class HAM:
    def __init__(self, root, df):
        self.root = root
        self.df = df
        
    def organise(self):
        index = self.df[self.df["dx"].isin(["df", "bcc", "vasc", "akiec"])].index
        self.df.drop(index=index, inplace=True)
        self.df["filename"] = self.df["image_id"] + ".jpg"
        self.df["filepath"] = self.root + "data/" + self.df["filename"]
        self.df["filepath_seg"] = self.root + "seg/" + self.df["filename"]
        self.df["filepath_mask"] = self.root + "mask/" + self.df["filename"]
        self.df.loc[self.df["dx"] == "nv", "label"] = "0"
        self.df.loc[self.df["dx"] == "mel", "label"] = "1"
        self.df.loc[self.df["dx"] == "bkl", "label"] = "2"
        #self.df.loc[self.df["dx"] == "df", "label"] = "3"
        #self.df.loc[self.df["dx"] == "bcc", "label"] = "4"
        #self.df.loc[self.df["dx"] == "vasc", "label"] = "5"
        #self.df.loc[self.df["dx"] == "akiec", "label"] = "6"
        print(f"Dataset Cleaned.")

    def resize_image(self, width, height):
        create_directory(f"{self.root}data")        
        for i, d in self.df.iterrows():
            input_image_path = f"{self.root}original/{d['filename']}"
            output_image_path = d["filepath"]
            resize(input_image_path, output_image_path, width, height)
            
        print(f"Resize completed")

    def resize_segmentation(self, width, height):
        create_directory(f"{self.root}seg")
        no_segmentation = []
        for i, d in self.df.iterrows():
            input_image_path = f"{self.root}ham_segmentation/{d['image_id']}_segmentation.png"
            if os.path.exists(input_image_path) == False:
                no_segmentation.append(d["image_id"])
                print(f"No segmentation: {input_image_path}")
            else:
                output_image_path = d["filepath_seg"]
                resize(input_image_path, output_image_path, width, height)
        if len(no_segmentation) != 0:
            index = self.df[self.df["image_id"].isin(no_segmentation)].index
            self.df.drop(index=index, inplace=True)
            
        print(f"Resize completed")
        
    def create_masked_image(self):
        create_directory(f"{self.root}mask")

        all_blacks = []
        for i, d in self.df.iterrows():
            filepath = d["filepath"]
            maskpath = d["filepath_seg"]
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
        
            if image.shape[:2] != mask.shape[:2]:
                print(filepath)
            else:
                skin = (mask == 0).astype(np.uint8) # 0:black, 1: white -> To change black: 1 means active
                masked_image = cv2.bitwise_and(image, image, mask=skin)
                if np.all(masked_image == 0):  # If all pixels in masked_image are [0, 0, 0]
                    all_blacks.append(d["image_id"])
                    print(f"All balck image: {d['image_id']}")
                masked_image = Image.fromarray(masked_image)
                masked_image.save(d["filepath_mask"])

        if len(all_blacks) != 0:
            index = self.df[self.df["image_id"].isin(all_blacks)].index
            self.df.drop(index=index, inplace=True)
        
        print("Created masked files.")

    def split_dataset(self, random_state):
        # balanced dataset
        groupsize = self.df.groupby(["label"]).size()
        print(f"Minimum: {groupsize.min()}")
        
        tmp1 = self.df[self.df["label"] == "0"].sample(n=groupsize.min(), random_state=random_state)
        tmp2 = self.df[self.df["label"] == "1"].sample(n=groupsize.min(), random_state=random_state)
        tmp3 = self.df[self.df["label"] == "2"].sample(n=groupsize.min(), random_state=random_state)
        df_balanced = pd.concat([tmp1, tmp2, tmp3])
        
        # split dataset
        create_directory(f"{self.root}dataframe")
        df_train, df_valid, df_test = split(df_balanced, random_state)
        df_train.to_csv(f"{self.root}/dataframe/df_train.csv", index=False)
        df_valid.to_csv(f"{self.root}/dataframe/df_valid.csv", index=False)
        df_test.to_csv(f"{self.root}/dataframe/df_test.csv", index=False)
        print("Files saved.")