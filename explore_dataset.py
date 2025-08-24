import os

# Correct path based on your screenshot
dataset_path = r"C:\Users\User\Downloads\archive\Bottle Images\Bottle Images"

# Actual folder names from your dataset
categories = ['Wine Bottle', 'Water Bottle', 'Soda Bottle', 'Plastic Bottles', 'Beer Bottles']

print("=== PRISM Bottle Dataset Exploration ===")
print(f"Dataset location: {dataset_path}\n")

total_images = 0

for category in categories:
    folder_path = os.path.join(dataset_path, category)
    if os.path.exists(folder_path):
        
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        image_count = len(image_files)
        total_images += image_count
        print(f"📁 {category}: {image_count} images")
        
        if image_files:
            print(f"   └── Sample files: {image_files[:3]}")
    else:
        print(f"❌ Folder not found: {folder_path}")

print(f"\n📊 Total images across all categories: {total_images}")
print(f"📂 Total categories: {len(categories)}")
