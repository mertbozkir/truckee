#!/usr/bin/env python3
"""
Minimal database script for Vehicle License Plate Detection System
"""

import sqlite3


def init_db():
    """Create simple database with just authorized plates"""
    conn = sqlite3.connect("src/data.db")
    cursor = conn.cursor()

    # Simple table with just plate numbers
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS plates (
            plate_number TEXT PRIMARY KEY
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Database created!")


def add_plate(plate_number):
    """Add a plate to database"""
    conn = sqlite3.connect("src/data.db")
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO plates (plate_number) VALUES (?)", (plate_number.upper(),)
        )
        conn.commit()
        print(f"✅ Added plate: {plate_number}")
    except sqlite3.IntegrityError:
        print(f"❌ Plate {plate_number} already exists!")
    conn.close()


def check_plate(plate_number):
    """Check if plate exists"""
    conn = sqlite3.connect("src/data.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT plate_number FROM plates WHERE plate_number = ?",
        (plate_number.upper(),),
    )
    result = cursor.fetchone()
    conn.close()
    return result is not None


def show_all_plates():
    """Show all plates in database"""
    conn = sqlite3.connect("src/data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT plate_number FROM plates")
    plates = cursor.fetchall()
    conn.close()

    if plates:
        print("📋 Authorized plates:")
        for plate in plates:
            print(f"  - {plate[0]}")
    else:
        print("📭 No plates found")


if __name__ == "__main__":
    init_db()

    # Add sample plates
    add_plate("34TK5678")
    add_plate("06TR9876")

    show_all_plates()

    # Test check
    print(
        f"\n🔍 Checking 34TK5678: {'✅ Authorized' if check_plate('34TK5678') else '❌ Not found'}"
    )
    print(
        f"🔍 Checking UNKNOWN: {'✅ Authorized' if check_plate('UNKNOWN') else '❌ Not found'}"
    )
