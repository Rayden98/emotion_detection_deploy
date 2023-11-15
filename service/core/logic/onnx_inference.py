import onnxruntime as rt
import cv2
import numpy as np

def emotions_detector(img_array):
    providers=['CUDAExecutionProvider']
    m_q = rt.InferenceSession("vit_keras.onnx", providers=providers)
    
    test_image = cv2.resize(test_image, (256, 256))
    im = np.float32(test_image)
    img_array = np.expand_dims(im, axis = 0)
    print(img_array.shape)
    
    onnx_pred = m_q.run(['dense'], {"input":img_array})
    print(np.argmax(onnx_pred[0][0]))
    
    emotion=""
    if np.argmax(onnx_pred[0][0]) ==0:
        emotion="angry"
    elif np.argmax(onnx_pred[0][0]) ==1:
        emotion="happy"
    else:
        emotion="sad"
        
    return {"emotion" : emotion}