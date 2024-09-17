from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io
import zipfile
import os
import shutil
import aiofiles

app = FastAPI()

processor = AutoImageProcessor.from_pretrained("cledoux42/Ethnicity_Test_v003")
model = AutoModelForImageClassification.from_pretrained("cledoux42/Ethnicity_Test_v003")

@app.post("/predict")
async def predict_ethnicity(file: UploadFile = File(...)):
    predictions = []

  
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)

    try:
   
        zip_path = os.path.join(temp_dir, file.filename)
        async with aiofiles.open(zip_path, 'wb') as out_file:
            content = await file.read() 
            await out_file.write(content)  

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        for root, dirs, files in os.walk(temp_dir):
            for filename in files:
                if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    try:
                        image_path = os.path.join(root, filename)
                        image = Image.open(image_path)

                        inputs = processor(images=image, return_tensors="pt")

                        with torch.no_grad():
                            outputs = model(**inputs)

             
                        logits = outputs.logits
                        probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

                        
                        class_labels = model.config.id2label 

                       
                        predicted_class_id = probabilities.index(max(probabilities))
                        predicted_label = class_labels[predicted_class_id]
                        predicted_probability = probabilities[predicted_class_id]

                    
                        predictions.append({
                            "filename": filename,
                            "predicted_ethnicity": predicted_label,
                            "probability": round(predicted_probability, 3)
                        })

                    except Exception as e:
                        predictions.append({
                            "filename": filename,
                            "error": str(e)
                        })

        shutil.rmtree(temp_dir)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=predictions)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
