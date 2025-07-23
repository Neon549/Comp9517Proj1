import os

"""configuration parameters"""
class Config:
    # data root
    PROJECT_ROOT = "usa_segmentation"
    MASKS_DIR = os.path.join(PROJECT_ROOT, "masks")
    RGB_IMAGES_DIR = os.path.join(PROJECT_ROOT, "RGB_images")
    NRG_IMAGES_DIR = os.path.join(PROJECT_ROOT, "NRG_images")
    USE_NRG = True
    INPUT_CHANNELS = 4 if USE_NRG else 3  # 4 channels (RGB + NIR)

    # model
    NUM_CLASSES = 2  # background
    INPUT_SIZE = (256, 256)  # input image size
    BACKBONE = 'resnet50'

    # train
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    TRAIN_SPLIT = 0.8
    RANDOM_SEED = 42

    # PSPNet
    PSP_SIZE = (1, 2, 3, 6)
    DEEP_FEATURES_SIZE = 256

    # enhancement
    USE_AUGMENTATION = True
    ROTATION_RANGE = 15
    FLIP_HORIZONTAL = True
    FLIP_VERTICAL = True
    BRIGHTNESS_RANGE = 0.2

    # output
    MODEL_SAVE_PATH = "models"
    RESULTS_SAVE_PATH = "results"
    LOG_DIR = "logs"

    # device
    DEVICE = "cuda"

    # channel
    USE_NRG = True  # if nrg
    INPUT_CHANNELS = 3 if USE_NRG else 3  # 3 channels

    @classmethod
    def create_directories(cls):
        """root path"""
        for path in [cls.MODEL_SAVE_PATH, cls.RESULTS_SAVE_PATH, cls.LOG_DIR]:
            os.makedirs(path, exist_ok=True)