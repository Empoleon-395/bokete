import torch
from transformers import BertTokenizer
from PIL import Image
import argparse
from catr.models import caption
from catr.datasets import coco, utils
from catr.configuration import Config
import os

# if torch.cuda.is_available:
#     device = "cuda:0"
# else:
#     device = "cpu"

device = "cpu"

class my_catr:
    def __init__(self,version = "v3"):
        checkpoint_path = None
        self.config = Config()

        if version == 'v1':
            self.model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True).to(device)
        elif version == 'v2':
            self.model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True).to(device)
        elif version == 'v3':
            self.model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True).to(device)
        else:
            print("Checking for checkpoint.")
            if checkpoint_path is None:
                raise NotImplementedError('No model to chose from!')
            else:
                if not os.path.exists(checkpoint_path):
                    raise NotImplementedError('Give valid checkpoint path')
                print("Found checkpoint! Loading!")
                model,_ = caption.build_model(self.config)
                print("Loading Checkpoint...")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(checkpoint['model'])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.start_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer._cls_token)
        self.end_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer._sep_token)

    def fit(self,image_path):
        self.image = Image.open(image_path)
        self.image = coco.val_transform(self.image)
        self.image = self.image.unsqueeze(0).to(device)

        self.caption, self.cap_mask = self.create_caption_and_mask(
        self.start_token, self.config.max_position_embeddings)
        self.caption = self.caption.to(device)
        self.cap_mask = self.cap_mask.to(device)

        output = self.evaluate()
        result = self.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        #result = tokenizer.decode(output[0], skip_special_tokens=True)
        return (result.capitalize())

    def create_caption_and_mask(self,start_token, max_length):
        caption_template = torch.zeros((1, max_length), dtype=torch.long)
        mask_template = torch.ones((1, max_length), dtype=torch.bool)

        caption_template[:, 0] = start_token
        mask_template[:, 0] = False

        return caption_template, mask_template

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        for i in range(self.config.max_position_embeddings - 1):
            predictions = self.model(self.image, self.caption, self.cap_mask)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)

            if predicted_id[0] == 102:
                return self.caption

            self.caption[:, i+1] = predicted_id[0]
            self.cap_mask[:, i+1] = False

        return self.caption