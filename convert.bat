python data/formats/convert_voc_to_yolo.py ^
  --voc-root E:\Dataset\SeaShips_SMD ^
  --output-root E:\Dataset\SeaShips_SMD_yolo ^
  --sets train_SMD train_SeaShips test_SeaShips test_SMD ^
  --classes-file E:\Dataset\SeaShips_SMD\classes.txt^
  --copy-images --keep-empty