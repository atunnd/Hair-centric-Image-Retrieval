import pandas as pd

def main(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    df_train_full_face = df_train
    df_test_full_face = df_test
    
    df_train_full_face['id'] = df_train.apply(lambda row: f"{row.id[:-9]}.jpg", axis=1)
    df_test_full_face['id'] = df_test.apply(lambda row: f"{row.id[:-9]}.jpg", axis=1)
    
    df_train_full_face.to_csv("data_train_full_face.csv", index=False)
    df_test_full_face.to_csv("data_test_full_face.csv", index=False)
    
    print(df_train_full_face.head())

if __name__=="__main__":
    main(train_path="data_train.csv", test_path="data_test.csv")