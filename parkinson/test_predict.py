from analyze import predict_parkinson
import json

res, err = predict_parkinson('uploads/voice.webm')
print('ERROR:', err)
print(json.dumps(res, indent=2, ensure_ascii=False))
