"""
Price Tracking Agent - Monitors product prices over time and provides trend analysis.

Features:
- Periodically queries the shopping agent to track prices
- Stores historical price data in SQLite database
- Provides statistics and trend analysis on demand
- Summarizes findings when tracking ends
- UI always accessible (tracking runs in background)
"""

import os
import sys
import sqlite3
import threading
import time
import queue
from datetime import datetime, timedelta
from typing import Optional, Dict

# Import the shopping agent
from main import search_product_prices, detect_product_category

# Database file
DB_FILE = "price_history.db"


class PriceTracker:
    """
    Agent that tracks product prices over time.
    UI always accessible - tracking runs in background threads.
    """
    
    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self.active_sessions: Dict[int, dict] = {}  # session_id -> {thread, stop_flag}
        self.output_queue = queue.Queue()  # For background thread output
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for price history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracking_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_query TEXT NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                interval_minutes INTEGER,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                retailer TEXT,
                product_url TEXT,
                base_price REAL,
                tax REAL,
                shipping REAL,
                total_price REAL,
                cashback_info TEXT,
                credit_card_info TEXT,
                FOREIGN KEY (session_id) REFERENCES tracking_sessions(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                retailer TEXT,
                old_price REAL,
                new_price REAL,
                change_percent REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES tracking_sessions(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _log(self, message: str):
        """Thread-safe logging."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.output_queue.put(f"[{timestamp}] {message}")
    
    def print_pending_output(self):
        """Print any pending output from background threads."""
        while not self.output_queue.empty():
            try:
                msg = self.output_queue.get_nowait()
                print(msg)
            except queue.Empty:
                break
    
    def start_tracking(self, product_query: str, interval_minutes: int = 60, duration_hours: Optional[int] = None) -> int:
        """
        Start tracking prices for a product in a background thread.
        Returns immediately - tracking continues in background.
        """
        # Create tracking session
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO tracking_sessions (product_query, interval_minutes, status)
            VALUES (?, ?, 'active')
        ''', (product_query, interval_minutes))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Create stop flag for this session
        stop_flag = threading.Event()
        
        # Start tracking in background thread
        tracking_thread = threading.Thread(
            target=self._tracking_loop,
            args=(session_id, product_query, interval_minutes, duration_hours, stop_flag),
            daemon=True,
            name=f"Tracker-{session_id}"
        )
        
        # Store thread info
        self.active_sessions[session_id] = {
            'thread': tracking_thread,
            'stop_flag': stop_flag,
            'product': product_query,
            'interval': interval_minutes
        }
        
        tracking_thread.start()
        
        return session_id
    
    def _tracking_loop(self, session_id: int, product_query: str, interval_minutes: int, 
                       duration_hours: Optional[int], stop_flag: threading.Event):
        """Background loop that periodically checks prices."""
        start_time = datetime.now()
        check_count = 0
        
        self._log(f"📈 Session #{session_id} started tracking: {product_query}")
        
        while not stop_flag.is_set():
            # Check if duration exceeded
            if duration_hours:
                elapsed = (datetime.now() - start_time).total_seconds() / 3600
                if elapsed >= duration_hours:
                    self._log(f"⏰ Session #{session_id}: Duration ({duration_hours}h) completed!")
                    break
            
            check_count += 1
            self._log(f"🔄 Session #{session_id}: Price check #{check_count}")
            
            try:
                # Query the shopping agent (this takes time)
                results = search_product_prices(product_query)
                
                # Parse and store results
                records_saved = self._store_price_data(session_id, results)
                self._log(f"✅ Session #{session_id}: Saved {records_saved} price records")
                
                # Check for significant price changes
                self._check_price_alerts(session_id)
                
            except Exception as e:
                self._log(f"❌ Session #{session_id} error: {e}")
            
            # Wait for next interval (check stop flag periodically)
            wait_seconds = interval_minutes * 60
            for _ in range(wait_seconds):
                if stop_flag.is_set():
                    break
                time.sleep(1)
        
        # Mark session as completed
        self._complete_session(session_id)
        self._log(f"⏹️ Session #{session_id} stopped")
        
        # Remove from active sessions
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
    
    def _complete_session(self, session_id: int):
        """Mark a session as completed in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE tracking_sessions
            SET end_time = CURRENT_TIMESTAMP, status = 'completed'
            WHERE id = ?
        ''', (session_id,))
        conn.commit()
        conn.close()
    
    def _store_price_data(self, session_id: int, results: str) -> int:
        """Parse results and store in database. Returns count of records saved."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        lines = results.split('\n')
        current_retailer = None
        current_data = {}
        records_saved = 0
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('**') and line.endswith('**'):
                if current_retailer and current_data.get('total_price'):
                    self._save_record(cursor, session_id, current_retailer, current_data)
                    records_saved += 1
                
                current_retailer = line.strip('*').strip()
                current_data = {}
            
            elif 'Base Price:' in line or 'base price:' in line.lower():
                price = self._extract_price(line)
                if price:
                    current_data['base_price'] = price
            
            elif 'Tax' in line and '$' in line:
                price = self._extract_price(line)
                if price:
                    current_data['tax'] = price
            
            elif 'Shipping:' in line or 'shipping:' in line.lower():
                if 'free' in line.lower():
                    current_data['shipping'] = 0.0
                else:
                    price = self._extract_price(line)
                    if price:
                        current_data['shipping'] = price
            
            elif 'TOTAL:' in line or 'Total:' in line:
                price = self._extract_price(line)
                if price:
                    current_data['total_price'] = price
            
            elif 'URL:' in line or 'url:' in line.lower():
                url = line.split(':', 1)[-1].strip()
                current_data['product_url'] = url
            
            elif 'Cashback' in line:
                current_data['cashback_info'] = line
            
            elif 'Credit Card' in line or '💳' in line:
                current_data['credit_card_info'] = line
        
        if current_retailer and current_data.get('total_price'):
            self._save_record(cursor, session_id, current_retailer, current_data)
            records_saved += 1
        
        conn.commit()
        conn.close()
        return records_saved
    
    def _save_record(self, cursor, session_id: int, retailer: str, data: dict):
        """Save a single price record."""
        cursor.execute('''
            INSERT INTO price_records 
            (session_id, retailer, product_url, base_price, tax, shipping, total_price, cashback_info, credit_card_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            retailer,
            data.get('product_url', ''),
            data.get('base_price', 0),
            data.get('tax', 0),
            data.get('shipping', 0),
            data.get('total_price', 0),
            data.get('cashback_info', ''),
            data.get('credit_card_info', '')
        ))
    
    def _extract_price(self, text: str) -> Optional[float]:
        """Extract price from text."""
        import re
        match = re.search(r'\$?([\d,]+\.?\d*)', text)
        if match:
            price_str = match.group(1).replace(',', '')
            try:
                return float(price_str)
            except:
                return None
        return None
    
    def _check_price_alerts(self, session_id: int):
        """Check for significant price changes and create alerts."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT retailer, total_price, timestamp
            FROM price_records
            WHERE session_id = ?
            ORDER BY timestamp DESC
        ''', (session_id,))
        
        latest_prices = {}
        for row in cursor.fetchall():
            retailer, price, timestamp = row
            if retailer not in latest_prices:
                latest_prices[retailer] = []
            latest_prices[retailer].append((price, timestamp))
        
        for retailer, prices in latest_prices.items():
            if len(prices) >= 2:
                new_price = prices[0][0]
                old_price = prices[1][0]
                
                if old_price > 0:
                    change_percent = ((new_price - old_price) / old_price) * 100
                    
                    if abs(change_percent) >= 5:
                        cursor.execute('''
                            INSERT INTO price_alerts (session_id, retailer, old_price, new_price, change_percent)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (session_id, retailer, old_price, new_price, change_percent))
                        
                        direction = "📉 DROPPED" if change_percent < 0 else "📈 INCREASED"
                        self._log(f"🚨 ALERT: {retailer} {direction} by {abs(change_percent):.1f}% (${old_price:.2f} → ${new_price:.2f})")
        
        conn.commit()
        conn.close()
    
    def stop_tracking(self, session_id: int) -> bool:
        """Stop tracking a session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['stop_flag'].set()
            return True
        return False
    
    def get_active_sessions(self) -> Dict[int, dict]:
        """Get all currently active tracking sessions."""
        return {
            sid: {
                'product': info['product'],
                'interval': info['interval'],
                'running': info['thread'].is_alive()
            }
            for sid, info in self.active_sessions.items()
        }
    
    def get_statistics(self, session_id: int):
        """Get current statistics for a tracking session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT product_query, start_time, status FROM tracking_sessions WHERE id = ?', (session_id,))
        session = cursor.fetchone()
        
        if not session:
            print(f"❌ Session {session_id} not found")
            return
        
        product_query, start_time, status = session
        
        print(f"\n{'='*60}")
        print(f"📊 STATISTICS - Session #{session_id} [{status.upper()}]")
        print(f"{'='*60}")
        print(f"Product: {product_query}")
        print(f"Started: {start_time}")
        
        cursor.execute('''
            SELECT 
                retailer,
                COUNT(*) as check_count,
                MIN(total_price) as min_price,
                MAX(total_price) as max_price,
                AVG(total_price) as avg_price
            FROM price_records
            WHERE session_id = ?
            GROUP BY retailer
            ORDER BY min_price ASC
        ''', (session_id,))
        
        rows = cursor.fetchall()
        if rows:
            print(f"\n{'Retailer':<20} {'Checks':<8} {'Min':<12} {'Max':<12} {'Avg':<12}")
            print("-" * 64)
            
            for row in rows:
                retailer, count, min_p, max_p, avg_p = row
                retailer_short = retailer[:18] + ".." if len(retailer) > 20 else retailer
                print(f"{retailer_short:<20} {count:<8} ${min_p:<10.2f} ${max_p:<10.2f} ${avg_p:<10.2f}")
        else:
            print("\nNo price data recorded yet.")
        
        cursor.execute('SELECT COUNT(*) FROM price_alerts WHERE session_id = ?', (session_id,))
        alert_count = cursor.fetchone()[0]
        print(f"\n🚨 Price alerts: {alert_count}")
        
        conn.close()
    
    def get_summary(self, session_id: int):
        """Generate comprehensive summary."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT product_query, start_time, end_time, interval_minutes, status
            FROM tracking_sessions WHERE id = ?
        ''', (session_id,))
        session = cursor.fetchone()
        
        if not session:
            print(f"❌ Session {session_id} not found")
            return
        
        product_query, start_time, end_time, interval, status = session
        
        print(f"\n{'='*60}")
        print(f"📋 SUMMARY - Session #{session_id} [{status.upper()}]")
        print(f"{'='*60}")
        print(f"📦 Product: {product_query}")
        print(f"📅 Period: {start_time} to {end_time or 'ongoing'}")
        print(f"⏱️ Interval: Every {interval} minutes")
        
        # Best deal
        cursor.execute('''
            SELECT retailer, MIN(total_price) as best_price, timestamp
            FROM price_records WHERE session_id = ?
            GROUP BY retailer ORDER BY best_price ASC LIMIT 1
        ''', (session_id,))
        
        best = cursor.fetchone()
        if best:
            print(f"\n🏆 BEST DEAL: {best[0]} at ${best[1]:.2f}")
        
        # Price trends
        cursor.execute('''
            SELECT retailer FROM price_records WHERE session_id = ? GROUP BY retailer
        ''', (session_id,))
        
        retailers = [row[0] for row in cursor.fetchall()]
        
        if retailers:
            print(f"\n📈 PRICE TRENDS:")
            for retailer in retailers:
                cursor.execute('''
                    SELECT total_price FROM price_records 
                    WHERE session_id = ? AND retailer = ?
                    ORDER BY timestamp ASC
                ''', (session_id, retailer))
                prices = [row[0] for row in cursor.fetchall()]
                
                if len(prices) >= 2:
                    first, last = prices[0], prices[-1]
                    if first > 0:
                        change = ((last - first) / first) * 100
                        trend = "📉" if change < 0 else "📈" if change > 0 else "➡️"
                        print(f"   {trend} {retailer}: ${first:.2f} → ${last:.2f} ({change:+.1f}%)")
        
        # Alerts
        cursor.execute('''
            SELECT retailer, old_price, new_price, change_percent, timestamp
            FROM price_alerts WHERE session_id = ? ORDER BY timestamp
        ''', (session_id,))
        
        alerts = cursor.fetchall()
        if alerts:
            print(f"\n🚨 ALERTS ({len(alerts)}):")
            for alert in alerts[-5:]:  # Show last 5
                retailer, old_p, new_p, change, ts = alert
                direction = "↓" if change < 0 else "↑"
                print(f"   {direction} {retailer}: ${old_p:.2f} → ${new_p:.2f} ({change:+.1f}%)")
        
        conn.close()
        print(f"{'='*60}\n")
    
    def list_sessions(self):
        """List all tracking sessions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, product_query, start_time, status, 
                   (SELECT COUNT(*) FROM price_records WHERE session_id = tracking_sessions.id) as records
            FROM tracking_sessions ORDER BY start_time DESC LIMIT 20
        ''')
        
        print(f"\n{'='*70}")
        print(f"📋 TRACKING SESSIONS")
        print(f"{'='*70}")
        print(f"{'ID':<5} {'Product':<25} {'Status':<12} {'Records':<10} {'Active':<8}")
        print("-" * 70)
        
        for row in cursor.fetchall():
            sid, query, start, status, records = row
            query_short = query[:23] + ".." if len(query) > 25 else query
            is_active = "✅" if sid in self.active_sessions else ""
            print(f"{sid:<5} {query_short:<25} {status:<12} {records:<10} {is_active:<8}")
        
        conn.close()


def interactive_menu():
    """Interactive menu - always accessible while tracking runs in background."""
    tracker = PriceTracker()
    
    print(f"\n{'='*60}")
    print("🛒 PRICE TRACKING AGENT")
    print("    Tracking runs in background - UI always available")
    print(f"{'='*60}")
    
    while True:
        # Print any pending output from background threads
        tracker.print_pending_output()
        
        # Show active sessions
        active = tracker.get_active_sessions()
        if active:
            print(f"\n🟢 Active tracking: {len(active)} session(s)")
            for sid, info in active.items():
                status = "running" if info['running'] else "stopping"
                print(f"   #{sid}: {info['product']} (every {info['interval']}min) [{status}]")
        
        print(f"\n{'-'*40}")
        print("1. Start new tracking")
        print("2. View statistics")
        print("3. View summary")
        print("4. Stop tracking")
        print("5. List all sessions")
        print("6. Check background updates")
        print("7. Exit")
        print(f"{'-'*40}")
        
        try:
            choice = input("Choice (1-7): ").strip()
        except EOFError:
            break
        
        # Print any new output
        tracker.print_pending_output()
        
        if choice == '1':
            product = input("Product to track (default: PlayStation 5): ").strip()
            if not product:
                product = "PlayStation 5"
            
            interval = input("Interval in minutes (default: 60): ").strip()
            interval = int(interval) if interval else 60
            
            duration = input("Duration in hours (empty = indefinite): ").strip()
            duration = int(duration) if duration else None
            
            session_id = tracker.start_tracking(product, interval, duration)
            print(f"\n✅ Started tracking session #{session_id}")
            print(f"   Product: {product}")
            print(f"   Interval: {interval} minutes")
            print(f"   Duration: {duration or 'indefinite'} hours")
            print(f"\n   Tracking runs in background - menu stays available!")
        
        elif choice == '2':
            session_id = input("Session ID (or 'all' for list): ").strip()
            if session_id.lower() == 'all':
                tracker.list_sessions()
            elif session_id:
                tracker.get_statistics(int(session_id))
        
        elif choice == '3':
            session_id = input("Session ID: ").strip()
            if session_id:
                tracker.get_summary(int(session_id))
        
        elif choice == '4':
            session_id = input("Session ID to stop: ").strip()
            if session_id:
                if tracker.stop_tracking(int(session_id)):
                    print(f"⏹️ Stopping session #{session_id}...")
                else:
                    print(f"❌ Session #{session_id} not active")
        
        elif choice == '5':
            tracker.list_sessions()
        
        elif choice == '6':
            print("\n📬 Checking for background updates...")
            tracker.print_pending_output()
            print("   Done!")
        
        elif choice == '7':
            # Stop all active sessions
            if tracker.active_sessions:
                print("\n⏹️ Stopping all active tracking sessions...")
                for sid in list(tracker.active_sessions.keys()):
                    tracker.stop_tracking(sid)
                time.sleep(2)  # Give threads time to stop
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    interactive_menu()
