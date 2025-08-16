# In your helper.py or add to existing code
import sqlite3
import numpy as np
from datetime import datetime

DB_FILE = "notulen.db"
SPEAKER_SIM_THRESHOLD = 0.85  # Increased threshold for single speaker


def init_speaker_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Check if table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='speaker_profiles'")
    table_exists = c.fetchone() is not None

    if table_exists:
        # Check if columns exist
        c.execute("PRAGMA table_info(speaker_profiles)")
        columns = [col[1] for col in c.fetchall()]

        # Add missing columns if needed
        if 'usage_count' not in columns:
            c.execute("ALTER TABLE speaker_profiles ADD COLUMN usage_count INTEGER DEFAULT 1")
    else:
        # Create new table with all columns
        c.execute('''
            CREATE TABLE speaker_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                embedding BLOB,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                usage_count INTEGER DEFAULT 1
            )
        ''')

    conn.commit()
    conn.close()


def save_speaker_embedding(name, emb: np.ndarray):
    emb_norm = emb / np.linalg.norm(emb)
    emb_bytes = emb_norm.astype(np.float32).tobytes()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Check if speaker exists
    c.execute("SELECT id FROM speaker_profiles WHERE name=?", (name,))
    row = c.fetchone()

    if row:
        # Update existing speaker with new embedding and increment count
        c.execute('''
            UPDATE speaker_profiles 
            SET embedding=?, last_seen=?, usage_count=usage_count+1 
            WHERE id=?
        ''', (emb_bytes, datetime.now(), row[0]))
    else:
        # Insert new speaker
        c.execute('''
            INSERT INTO speaker_profiles (name, embedding) 
            VALUES (?,?)
        ''', (name, emb_bytes))

    conn.commit()
    conn.close()


def find_speaker_by_embedding(emb: np.ndarray):
    emb_norm = emb / np.linalg.norm(emb)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    try:
        # Check if usage_count column exists
        c.execute("PRAGMA table_info(speaker_profiles)")
        columns = [col[1] for col in c.fetchall()]
        has_usage_count = 'usage_count' in columns

        if has_usage_count:
            c.execute('''
                SELECT id, name, embedding, usage_count 
                FROM speaker_profiles 
                ORDER BY last_seen DESC, usage_count DESC
            ''')
        else:
            c.execute('''
                SELECT id, name, embedding, 1 as usage_count 
                FROM speaker_profiles 
                ORDER BY last_seen DESC
            ''')

        best_id, best_name, best_sim = None, None, -1

        for row in c.fetchall():
            id_, name, emb_bytes, count = row
            ref = np.frombuffer(emb_bytes, dtype=np.float32)
            ref = ref / np.linalg.norm(ref)
            sim = np.dot(emb_norm, ref)

            # Weight by usage count if available
            weighted_sim = sim * (1 + 0.1 * min(count, 10)) if has_usage_count else sim

            if weighted_sim > best_sim:
                best_sim, best_id, best_name = weighted_sim, id_, name

        if best_sim >= SPEAKER_SIM_THRESHOLD:
            # Update last_seen
            if has_usage_count:
                c.execute('''
                    UPDATE speaker_profiles 
                    SET last_seen=?, usage_count=usage_count+1 
                    WHERE id=?
                ''', (datetime.now(), best_id))
            else:
                c.execute('''
                    UPDATE speaker_profiles 
                    SET last_seen=?
                    WHERE id=?
                ''', (datetime.now(), best_id))
            conn.commit()
            return best_name

        return None

    finally:
        conn.close()


def merge_speakers(main_speaker, secondary_speaker):
    """Menggabungkan semua ucapan secondary_speaker ke main_speaker"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    try:
        # 1. Update di database
        c.execute("""
            UPDATE transkrip 
            SET pembicara = ? 
            WHERE pembicara = ?
        """, (main_speaker, secondary_speaker))

        # 2. Update embedding jika ada
        c.execute("""
            UPDATE speaker_profiles 
            SET name = ?
            WHERE name = ?
        """, (main_speaker, secondary_speaker))

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

