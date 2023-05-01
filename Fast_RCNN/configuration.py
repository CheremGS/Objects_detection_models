"""
Configuration dictionaries for train parameters and for dataaset preporation.
There is no need to add a unit to the number of classes for an additional background class,
since this is taken into account when creating the model.
"""

config_dataset = {
                  "path_image_dir": r"C:\Users\ITC-Admin\PycharmProjects\Detection_military\datasets\airplanes\JPEGImages",
                  "path_annots_dir": r"C:\Users\ITC-Admin\PycharmProjects\Detection_military\datasets\airplanes\Annotations\Horizontal Bounding Boxes",
                  "test_train_proportion": 0.1, # range (0, 1)
                  "batch_size": 1, # range (1, inf)
                  "image_shape": [768, 768, 3], # (Height, Width, Number of channels)
                  "num_classes": 20 # range (1, inf)
                 }

config_train = {"n_epochs": 50, # range (1, inf)
                "patience_EarlyStopping": 7, # range (1, inf)
                "threshold_EarlyStopping": 1e-4, # range (0, 1)
                "patience_lr_Scheduler": 3 # range (1, inf)
                }