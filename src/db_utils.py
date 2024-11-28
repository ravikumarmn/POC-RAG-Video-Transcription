import sqlite3
import json
from typing import Dict, Optional

class TranscriptDB:
    def __init__(self, db_path: str = "data/transcripts.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create transcripts table
        c.execute('''
            CREATE TABLE IF NOT EXISTS transcripts
            (video_id TEXT PRIMARY KEY,
             title TEXT,
             url TEXT,
             transcript_data TEXT,
             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        ''')
        
        conn.commit()
        conn.close()

    def upsert_transcript(self, video_id: str, title: str, url: str, transcript_data: Dict) -> bool:
        """Insert or update transcript data"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('''
                INSERT OR REPLACE INTO transcripts (video_id, title, url, transcript_data)
                VALUES (?, ?, ?, ?)
            ''', (video_id, title, url, json.dumps(transcript_data)))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error upserting transcript: {e}")
            return False

    def get_transcript(self, video_id: str) -> Optional[Dict]:
        """Retrieve transcript data for a video"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('SELECT transcript_data FROM transcripts WHERE video_id = ?', (video_id,))
            result = c.fetchone()
            
            conn.close()
            
            if result:
                return json.loads(result[0])
            return None
        except Exception as e:
            print(f"Error retrieving transcript: {e}")
            return None

    def get_all_videos(self):
        """Get all stored video information"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('SELECT video_id, title, url FROM transcripts')
            videos = c.fetchall()
            
            conn.close()
            
            return [{'video_id': v[0], 'title': v[1], 'url': v[2]} for v in videos]
        except Exception as e:
            print(f"Error retrieving videos: {e}")
            return []
