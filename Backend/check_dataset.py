import os

dataset_path = "RawData/Quantum Satellite Image Classifier"

classes = os.listdir(dataset_path)

print("Dataset Classes:")
print("\n")


for class_name in classes:

    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):

        images = os.listdir(class_path)

        print(class_name, ":", len(images), "images")

print("\nTotal Classes:", len(classes))