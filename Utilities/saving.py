from keras.models import load_model, save_model
import h5py

classes = ["BTR-82", "URAL_gruz", "Kamaz_gruz", "URAL_kung", "Soldier", "Tiger", "BMP-3", "T-72", "KRAZ_FUEL"]
model_path = r"C:\Users\ITC-Admin\PycharmProjects\itc_detector\copy_with_pad\oko_mavic_classification_leakyrelu.h5"
model = load_model(model_path)

# saving model
file_name = "model_with_classes.h5"
save_model(model, file_name, overwrite=True)
f = h5py.File(file_name, mode='a')
f.attrs['list_classes'] = classes
f.close()

# loading model
model = load_model(file_name, custom_objects=None)
f = h5py.File(file_name, mode='r')
meta_data = None
if 'list_classes' in f.attrs:
    meta_data = f.attrs.get('list_classes')
f.close()