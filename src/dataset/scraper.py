import os
import json
import logging
from typing import List, Dict, Optional
import yt_dlp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AlgerianAudioScraper:
    """
    Automated Scraper to search and download public audio content (YouTube, Podcasts, Short clips)
    targeting Algerian Darja (Algiers dialect focus).
    """

    def __init__(self, output_dir: str = "data/raw_audio"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def download_youtube_audio(self, url: str, item_id: Optional[str] = None) -> Optional[Dict[str, str]]:
        """
        Downloads audio from a YouTube URL and converts to standard WAV format.
        """
        if not item_id:
            item_id = "".join(c for c in url if c.isalnum())[-12:]

        out_template = os.path.join(self.output_dir, f"{item_id}.%(ext)s")

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': out_template,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }

        try:
            logger.info(f"Downloading audio from: {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get("title", "Unknown Title")
                duration = info.get("duration", 0)

            expected_wav = os.path.join(self.output_dir, f"{item_id}.wav")
            metadata = {
                "id": item_id,
                "url": url,
                "title": title,
                "duration": duration,
                "filepath": expected_wav if os.path.exists(expected_wav) else None,
                "dialect_region": "Algiers",
                "language": "Algerian Darja"
            }
            logger.info(f"Successfully downloaded: {title} ({duration}s)")
            return metadata
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None

    def search_algerian_darja_sources(self, query_keywords: List[str], max_results: int = 5) -> List[Dict[str, str]]:
        """
        Searches for Algerian Darja content on YouTube based on targeted search queries.
        """
        found_items = []
        for keyword in query_keywords:
            search_query = f"ytsearch{max_results}:{keyword} algerie darja podcast speech"
            logger.info(f"Searching for query: '{search_query}'")
            ydl_opts = {
                'quiet': True,
                'extract_flat': True,
                'skip_download': True,
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    res = ydl.extract_info(search_query, download=False)
                    if 'entries' in res:
                        for entry in res['entries']:
                            found_items.append({
                                "id": entry.get("id"),
                                "title": entry.get("title"),
                                "url": f"https://www.youtube.com/watch?v={entry.get('id')}",
                                "query_used": keyword
                            })
            except Exception as e:
                logger.error(f"Error searching query '{keyword}': {e}")
        return found_items

if __name__ == "__main__":
    scraper = AlgerianAudioScraper(output_dir="data/raw_audio")
    sample_queries = ["بودكاست دزيري", "حديث العاصمة الجزائر"]
    logger.info("Scraper script operational.")
