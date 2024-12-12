import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

'''
# Transformations à appliquer
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Liste des vraies valeurs prdtypecode
prdtypecode_list = ['10', '1140', '1160', '1180', '1280', '1281', '1300', '1301', 
                    '1302', '1320', '1560', '1920', '1940', '2060', '2220', 
                    '2280', '2403', '2462', '2522', '2582', '2583', '2585', 
                    '2705', '2905', '40', '50', '60']
prdtypecode_sorted = ['10', '40', '50', '60', '1140', '1160', '1180', '1280', '1281', '1300', '1301', '1302', '1320', 
                      '1560', '1920', '1940', '2060', '2220', '2280', '2403', '2462', '2522', '2582', '2583', '2585', 
                      '2705', '2905']
'''

# Classe de l'objet final (transfo + model)
class ViTPipeline:
    def __init__(self, model, transform, prdtypecode_list, prdtypecode_sorted):
        self.model = model
        self.transform = transform
        self.prdtypecode_list = prdtypecode_list
        self.prdtypecode_sorted = prdtypecode_sorted
        self.model.eval()

    def predict(self, image_path):
        # Ouverture de l'image
        img = Image.open(image_path)
        
        # Transformation de l'image
        img_transformed = self.transform(img).unsqueeze(0)
        
        # Prédiction du modèle
        with torch.no_grad():
            output = self.model(img_transformed)
            probabilities = F.softmax(output, dim = 1).squeeze()
        
        sorted_probabilities = self.reorganize_probabilities(probabilities)
        _, predicted_class = torch.max(sorted_probabilities, 0)
        predicted_prdtypecode = self.prdtypecode_sorted[predicted_class]

        return predicted_prdtypecode, sorted_probabilities
    
    def reorganize_probabilities(self, probabilities):
        """Réorganise les probabilités selon l'ordre de prdtypecode_sorted"""
        # Mapping des indices entre prdtypecode_list et prdtypecode_sorted
        idx_mapping = [self.prdtypecode_list.index(code) for code in self.prdtypecode_sorted]
        # Réorganiser les probabilités selon le nouvel ordre
        sorted_probabilities = probabilities[idx_mapping]
        return sorted_probabilities