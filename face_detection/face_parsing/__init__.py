from transformers import SegformerImageProcessor
from transformers import SegformerForSemanticSegmentation
from torch import nn


class FaceSegmentation:
    """
    Wrapper class for jonathandinu's face-parsing model for face segmentation.
    Huggingface: https://huggingface.co/jonathandinu/face-parsing
    """

    LABELS = {
        0: "background",
        1: "skin",
        2: "nose",
        3: "eyeglasses",
        4: "left eye",
        5: "right eye",
        6: "left eyebrow",
        7: "right eyebrow",
        8: "left ear",
        9: "right ear",
        10: "mouth",
        11: "upper lip",
        12: "lower lip",
        13: "hair",
        14: "hat",
        15: "earring",
        16: "necklace",
        17: "neck",
        18: "clothing",
    }

    def __init__(self, device="cpu"):
        self.device = device
        self.processor = SegformerImageProcessor.from_pretrained(
            "jonathandinu/face-parsing"
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "jonathandinu/face-parsing"
        )
        self.model.to(device)

    def get_mask(self, image):
        "Return the segmentation mask from an image"
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Resize output to match input image shape
        upsampled_logits = nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False  # H x W
        )
        # Get label masks
        labels = upsampled_logits.argmax(dim=1)[0]
        # Move mask to CPU
        labels = labels.cpu().numpy()
        return labels
