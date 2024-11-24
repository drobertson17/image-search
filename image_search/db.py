import logging
import os
from dotenv import load_dotenv

from sqlalchemy import create_engine, insert, exists
from sqlalchemy import Column, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv()
logging.basicConfig()
engine = create_engine(os.getenv("IMAGE_SEARCH_DATABASE"))
Base = declarative_base()

class ImageDescription(Base):
    __tablename__ = "image_description"

    id = Column(String, primary_key=True)
    full_path = Column(String)
    base_dir = Column(String)
    file_name = Column(String)
    file_type = Column(String)
    long_desc = Column(String)
    short_desc = Column(String)
    keywords = Column(String)
    title = Column(String)
    image_classification = Column(String)
    predominant_color = Column(String)
    natural_landscape = Column(Boolean, default=False)
    building_structure = Column(Boolean, default=False)
    selfie = Column(Boolean, default=False)
    friends_or_family = Column(Boolean, default=False)
    ignore = Column(Boolean, default=False)


# Create all tables in the engine
Base.metadata.create_all(engine)

# Create a session to interact with the database
Session = sessionmaker(bind=engine)

class ImageDBService:
    def __init__(self):
        self.logger = logging.getLogger("db")
        self.logger.setLevel(logging.DEBUG)
        try:
            self.db = Session()
        except Exception as e:
            self.logger.warning(f"ERROR: {e}")

    def image_exists(self, full_path: str) -> bool:
        return self.db.query(exists().where(ImageDescription.full_path == full_path)).scalar()

    def add_image_description(self, img_desc: dict):
        if not self.image_exists(img_desc["full_path"]):
            q = insert(ImageDescription).values(img_desc)
            self.db.execute(q)
            self.db.commit()