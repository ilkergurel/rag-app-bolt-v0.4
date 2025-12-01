import os
from pymongo import MongoClient
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DatabaseBookService:
    def __init__(self):
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        database_name = os.getenv("DATABASE_NAME", "bookdb")
        collection_name = os.getenv("COLLECTION_NAME", "books")

        try:
            self.client = MongoClient(mongodb_uri)
            self.db = self.client[database_name]
            self.collection = self.db[collection_name]
            logger.info(f"✅ MongoDB connected: {database_name}.{collection_name}")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise

    def query_books(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query books from MongoDB based on extracted parameters.
        Returns list of books with metadata.
        """
        query = {}

        # Build MongoDB query
        if "author" in parameters and parameters["author"]:
            query["author"] = {"$regex": parameters["author"], "$options": "i"}

        if "name" in parameters and parameters["name"]:
            query["name"] = {"$regex": parameters["name"], "$options": "i"}

        if "year_start" in parameters or "year_end" in parameters:
            year_query = {}
            if "year_start" in parameters:
                year_query["$gte"] = parameters["year_start"]
            if "year_end" in parameters:
                year_query["$lte"] = parameters["year_end"]
            query["year"] = year_query

        if "keywords" in parameters and parameters["keywords"]:
            # This creates an "OR" logic with Regex for each keyword provided
            keyword_conditions = []
            for kw in parameters["keywords"]:
                keyword_conditions.append({"keywords": {"$regex": kw, "$options": "i"}})
            
            # $or means: If ANY of these regex patterns match, return the book
            if keyword_conditions:
                query["$or"] = keyword_conditions

        logger.info(f"MongoDB Query: {query}")

        try:
            # Limit to 100 results
            #results = list(self.collection.find(query, {"_id": 0}).limit(100))
            results = list(self.collection.find(query, {"_id": 0})) # !!! limit to 100 removed


            logger.info(f"Found {len(results)} books")

            formatted_results = []
            for book in results:
                # Ensure keywords is always a list
                keywords_raw = book.get("keywords", [])
                if isinstance(keywords_raw, str):
                    # If it's a string, wrap it in a list
                    keywords = [keywords_raw] if keywords_raw else []
                elif isinstance(keywords_raw, list):
                    keywords = keywords_raw
                else:
                    keywords = []

                formatted_results.append({
                    "name": book.get("name", "Unknown"),
                    "path": book.get("path", ""),
                    "author": book.get("author", "Unknown"),
                    "year": book.get("year", "N/A"),
                    "keywords": keywords
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Database query error: {e}")
            return []

    def close(self):
        """Close MongoDB connection"""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("MongoDB connection closed")
