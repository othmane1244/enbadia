import os

# Compter les images
train_dir = r"C:\Users\user\Documents\projet\fine-tuning 1\data\train\images"
valid_dir = r"C:\Users\user\Documents\projet\fine-tuning 1\data\valid\images"

print(f"Train: {len(os.listdir(train_dir))} images")
print(f"Valid: {len(os.listdir(valid_dir))} images")

# Vérifier les labels
labels_dir = r"C:\Users\user\Documents\projet\fine-tuning 1\data\train\labels"
files = os.listdir(labels_dir)

# Compter les classes
class_0 = 0  # fall
class_1 = 0  # autre (si existe)

for f in files:
    with open(os.path.join(labels_dir, f)) as file:
        for line in file:
            cls = int(line.split()[0])
            if cls == 0:
                class_0 += 1
            else:
                class_1 += 1

print(f"\nAnnotations:")
print(f"  Class 0 (fall): {class_0}")
print(f"  Class 1 (autre): {class_1}")