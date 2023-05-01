import onnx2tf
import onnx
import json

# d_ = {"format_version": 1,
#       "operations": [
#           {"op_name": "/model.24/Add",
#            "param_target": "inputs",
#            "param_name": "Mul_output_0",
#            "pre_process_transpose_perm": [0, 4, 2, 3, 1]}
#       ]
#       }

path_json = 'corr.json'
path_model = r"C:\Users\ITC-Admin\PycharmProjects\Detection_military\detector_small_random_objs\yolov5\runs\train\exp1\weights\oko1024_rectangle_fork_classif.onnx"

onnx_model = onnx.load(path_model)
onnx.checker.check_model(onnx_model)

# with open(path_json, "w") as wf:
#     json.dump(d_, wf)


a = onnx2tf.convert(input_onnx_file_path = path_model,
                    #overwrite_input_shape=["data:1,1,1024,1024"],
                    #output_h5=True,
                    #output_weights = True,
                    #param_replacement_file=path_json,
                    batch_size=1
                    )