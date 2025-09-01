from constants import ENV_PATH
import os
from dotenv import load_dotenv

print("ENV_PATH =", ENV_PATH)
print("Exists?", os.path.exists(ENV_PATH))
load_dotenv(ENV_PATH)
