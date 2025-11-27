"""
study_buddy_agent_humanized.py

The Friendly Study Buddy Agent!
This script helps you stop stressing by making a realistic, personalized
study schedule for all your big exams and projects. It tries to spread
the work out, warns you if you're taking on too much, and saves the plan.

No fancy external libraries needed—just pure Python.
"""

from __future__ import annotations
import logging
import sys
import json
import csv
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date
from pathlib import Path

# --- 📚 The Friendly Logbook ---
def set_up_our_logbook(name: str = "study_buddy", level: int = logging.INFO) -> logging.Logger:
    """Sets up a nice logger so we can see what's happening."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    # A cleaner format for humans
    formatter_style = "[%(asctime)s] 🌟 %(levelname)s: %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(formatter_style))
    logger.addHandler(handler)
    return logger

log = set_up_our_logbook("study_buddy")

# --- ⏰ Time Management Ninja Tools ---
class TimeGuru:
    """A collection of static methods to help us calculate days and distribute hours evenly."""

    @staticmethod
    def count_days_inclusive(start_day: date, end_day: date) -> int:
        """How many days are there from start to end, including both days?"""
        delta_days = (end_day - start_day).days
        # We always want at least 1 day if the dates are valid
        return max(0, delta_days) + 1

    @staticmethod
    def spread_the_load(required_total_hours: float, total_days_available: int) -> List[float]:
        """
        Takes the total study hours and spreads them as evenly as possible 
        across the available days. We handle the small rounding differences here!
        Returns a list of hours assigned to each day.
        """
        if total_days_available <= 0:
            return []

        base_daily_share = float(required_total_hours) / total_days_available
        daily_hours_list = [round(base_daily_share, 2) for _ in range(total_days_available)]

        # Little adjustment loop to fix minor floating point errors
        rounding_difference = round(required_total_hours - sum(daily_hours_list), 2)
        day_index = 0
        
        while abs(rounding_difference) >= 0.01:
            # Add or subtract the smallest amount (0.01) to one day at a time
            adjustment = 0.01 if rounding_difference > 0 else -0.01
            daily_hours_list[day_index % total_days_available] = round(daily_hours_list[day_index % total_days_available] + adjustment, 2)
            
            rounding_difference = round(required_total_hours - sum(daily_hours_list), 2)
            day_index += 1

        # NOTE: We skip rebalancing with daily_cap here; that's the Agent's job later!
        return [round(h, 2) for h in daily_hours_list]

# --- 📁 The File Keeper Service ---
class ScheduleFileKeeper:
    """Handles saving the final schedule data in human-readable formats (JSON and CSV)."""

    def __init__(self, output_folder: str = "schedules"):
        self.output_directory = Path(output_folder)
        # Make sure the folder exists!
        self.output_directory.mkdir(parents=True, exist_ok=True)
        log.info("Schedules will be saved in the '%s' folder.", output_folder)

    def save_as_json(self, file_name: str, complete_data: Any) -> Path:
        """Saves the full, structured data in a JSON file."""
        path = self.output_directory / f"{file_name}.json"
        try:
            path.write_text(json.dumps(complete_data, indent=2), encoding="utf-8")
            log.info("🎉 Success! Saved detailed JSON schedule -> %s", path)
        except IOError as e:
            log.error("Failed to save JSON file: %s", e)
        return path

    def save_as_csv(self, file_name: str, daily_rows: List[Dict[str, Any]]) -> Path:
        """Saves a simple, spreadsheet-friendly CSV version."""
        path = self.output_directory / f"{file_name}.csv"
        if not daily_rows:
            path.write_text("date,total_hours\n", encoding="utf-8")
            return path
        
        # Get all column names (Date, Total Hours, plus every Subject name)
        all_column_names = list(daily_rows[0].keys()) 
        
        try:
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=all_column_names)
                writer.writeheader()
                writer.writerows(daily_rows)
            log.info("🎉 Success! Saved CSV summary table -> %s", path)
        except IOError as e:
            log.error("Failed to save CSV file: %s", e)
        return path

# --- 🧠 The Master Planner Agent ---
class StudyPlannerAgent:
    """
    I am the Agent! My job is to take your subject list and turn it into a
    day-by-day plan that minimizes stress and meets all your deadlines.
    """

    def __init__(self, daily_study_limit: Optional[float] = None, plan_starts_on: Optional[date] = None):
        """Initializes the planner with rules and starting point."""
        self.daily_max = daily_study_limit
        self.today = plan_starts_on or date.today()
        self.time_tool = TimeGuru()
        log.info("Planner initialized. Starting today: %s. Daily study max: %s", self.today, self.daily_max or "None")

    def create_plan(self, study_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """The main method: process requests and build the schedule."""
        
        normalized_tasks = []
        problem_warnings: List[str] = []

        # 1. Preprocess: Clean up inputs, check dates, and calculate the planning window
        for request in study_requests:
            subj_name = request.get("subject")
            hours_needed = float(request.get("total_hours", 0))
            deadline_string = request.get("deadline")
            priority_level = int(request.get("priority", 1))
            
            if not subj_name or hours_needed <= 0 or not deadline_string:
                problem_warnings.append(f"Skipping invalid task: missing subject, hours, or deadline in {request}")
                continue
            
            try:
                deadline_date = datetime.strptime(deadline_string, "%Y-%m-%d").date()
            except Exception:
                problem_warnings.append(f"🛑 Error: Invalid date format for '{subj_name}': {deadline_string}. Please use YYYY-MM-DD.")
                continue
            
            days_until_deadline = self.time_tool.count_days_inclusive(self.today, deadline_date)
            
            if days_until_deadline <= 0:
                # If the deadline is today or passed, we have 1 day (today) to do it all!
                days_until_deadline = 1
                problem_warnings.append(f"⚠️ Warning: Deadline for '{subj_name}' is today or passed ({deadline_date}). Squeezing into 1 day.")

            normalized_tasks.append({
                "subject": subj_name,
                "total_hours": round(hours_needed, 2),
                "deadline": deadline_date,
                "days_available": days_until_deadline,
                "priority": priority_level
            })
        
        if not normalized_tasks:
            log.warning("No valid tasks found. Plan is empty.")
            return {"metadata": {"start_date": str(self.today)}, "schedule": {}, "subjects": {}, "warnings": problem_warnings}

        # Determine the full range of the schedule
        latest_finish_date = max(item["deadline"] for item in normalized_tasks)
        total_planning_span = self.time_tool.count_days_inclusive(self.today, latest_finish_date)
        all_dates_in_plan = [self.today + timedelta(days=i) for i in range(total_planning_span)]

        # Set up our daily calendar buckets
        daily_schedule_buckets: Dict[date, Dict[str, float]] = {d: {} for d in all_dates_in_plan}
        subject_data_lookup: Dict[str, Dict[str, Any]] = {item["subject"]: item for item in normalized_tasks}
        
        # 2. Initial Allocation: Fill the calendar based on urgency and priority
        
        # Sort by: 1) Earliest deadline, 2) Highest priority (negative sign makes higher number sort first)
        tasks_to_allocate = sorted(normalized_tasks, key=lambda x: (x["deadline"], -x["priority"]))
        
        log.info("Starting allocation for %d tasks...", len(tasks_to_allocate))

        for task in tasks_to_allocate:
            subject = task["subject"]
            total_required = task["total_hours"]
            deadline = task["deadline"]
            
            # Find all the dates *before or on* this subject's deadline
            available_days_for_subject = [d for d in all_dates_in_plan if d <= deadline]
            
            # Get the initial even split using the TimeGuru
            initial_daily_split = self.time_tool.spread_the_load(total_required, len(available_days_for_subject))
            
            # Place the split hours into our calendar buckets
            for day, hours_for_subject in zip(available_days_for_subject, initial_daily_split):
                if hours_for_subject > 0.01:
                    daily_schedule_buckets[day].setdefault(subject, 0.0)
                    daily_schedule_buckets[day][subject] = round(daily_schedule_buckets[day][subject] + hours_for_subject, 2)

        # 3. Load Balancing: Adjust the schedule to respect the daily max (if set)

        if self.daily_max is not None:
            log.info("Daily cap of %.2f hours is active. Starting load balancing...", self.daily_max)
            
            for index, current_day in enumerate(all_dates_in_plan):
                total_for_day = round(sum(daily_schedule_buckets[current_day].values()), 2)
                
                if total_for_day > self.daily_max + 1e-9: # Check for slight float error
                    excess_hours = round(total_for_day - self.daily_max, 2)
                    log.info("Day %s is overloaded by %.2f hours. Rebalancing...", current_day.isoformat(), excess_hours)
                    
                    # Try to push the excess hours to later days
                    
                    # Take the highest-priority subjects first to move (this is a heuristic choice)
                    subjects_to_move = sorted(daily_schedule_buckets[current_day].items(), 
                                              key=lambda item: subject_data_lookup[item[0]]["priority"], 
                                              reverse=True)
                    
                    for subj_name, subject_hours_today in subjects_to_move:
                        if excess_hours <= 0:
                            break
                        
                        # How much of this subject's time should we try to move?
                        amount_to_move = min(subject_hours_today, excess_hours)
                        
                        moved_successfully = False
                        
                        # Look at all days *after* today
                        for future_day in all_dates_in_plan[index + 1:]:
                            
                            # CRUCIAL CHECK: Can't move past the subject's deadline!
                            if future_day > subject_data_lookup[subj_name]["deadline"]:
                                continue
                            
                            # Check capacity of the future day
                            future_day_total = round(sum(daily_schedule_buckets[future_day].values()), 2)
                            capacity = round(self.daily_max - future_day_total, 2)
                            
                            if capacity > 0:
                                hours_to_transfer = min(amount_to_move, capacity)
                                
                                # 1. Remove from the current day
                                daily_schedule_buckets[current_day][subj_name] = round(daily_schedule_buckets[current_day].get(subj_name, 0.0) - hours_to_transfer, 2)
                                if daily_schedule_buckets[current_day][subj_name] <= 0.0001:
                                    del daily_schedule_buckets[current_day][subj_name]

                                # 2. Add to the future day
                                daily_schedule_buckets[future_day].setdefault(subj_name, 0.0)
                                daily_schedule_buckets[future_day][subj_name] = round(daily_schedule_buckets[future_day][subj_name] + hours_to_transfer, 2)
                                
                                # 3. Update remaining excess
                                excess_hours = round(excess_hours - hours_to_transfer, 2)
                                amount_to_move = round(amount_to_move - hours_to_transfer, 2)
                                moved_successfully = True
                                
                                if amount_to_move <= 0: # Done moving this subject's time
                                    break
                        
                        if amount_to_move > 0:
                            log.debug("Couldn't move all %.2f hours for %s. Leaving them as overload.", amount_to_move, subj_name)
                    
                    # Final check for warnings if we still have excess hours
                    if excess_hours > 0.01:
                         problem_warnings.append(f"🛑 Major Overload on {current_day.isoformat()}: Scheduled {total_for_day:.2f}h which is {excess_hours:.2f}h over the cap of {self.daily_max}h. Could not rebalance!")

        # 4. Final Summary Compilation
        
        schedule_output: Dict[str, Dict[str, float]] = {}
        csv_friendly_rows: List[Dict[str, Any]] = []
        
        # We need a list of all subjects for consistent CSV columns
        all_subjects_for_columns = list(subject_data_lookup.keys())

        for d in all_dates_in_plan:
            day_map = daily_schedule_buckets.get(d, {})
            
            # Format for JSON output
            schedule_output[str(d)] = {k: round(v, 2) for k, v in day_map.items()}
            
            total_day_hours = round(sum(day_map.values()), 2)
            
            # Format for CSV output
            row_for_csv = {"date": str(d), "total_hours": total_day_hours}
            for subject_column in all_subjects_for_columns:
                row_for_csv[subject_column] = day_map.get(subject_column, 0.0)
            csv_friendly_rows.append(row_for_csv)

        # Subject Summary
        final_subject_summary = {}
        for s, info in subject_data_lookup.items():
            total_scheduled = sum(day_map.get(s, 0.0) for day_map in daily_schedule_buckets.values())
            
            final_subject_summary[s] = {
                "requested_hours": info["total_hours"],
                "scheduled_hours": round(total_scheduled, 2),
                "deadline": str(info["deadline"]),
                "priority": info["priority"]
            }
            
            if final_subject_summary[s]["scheduled_hours"] + 0.01 < final_subject_summary[s]["requested_hours"]:
                 problem_warnings.append(f"❌ UNMET GOAL: Subject '{s}' only scheduled {final_subject_summary[s]['scheduled_hours']:.2f} of the required {final_subject_summary[s]['requested_hours']:.2f} hours by the deadline!")

        return {
            "metadata": {
                "generated_on": datetime.utcnow().isoformat() + "Z",
                "start_day": str(self.today),
                "latest_day": str(latest_finish_date),
                "daily_max_hours": self.daily_max
            },
            "schedule": schedule_output,
            "day_rows_for_csv": csv_friendly_rows, # Renamed for clarity
            "subjects_summary": final_subject_summary, # Renamed for clarity
            "warnings": problem_warnings
        }

# --- 🗣️ The Friendly Assistant ---
class HumanAssistant:
    """Takes the complex plan and presents it nicely to the user."""

    @staticmethod
    def show_me_the_plan(plan_result: Dict[str, Any], number_of_days_to_show: int = 5) -> None:
        """Prints a user-friendly summary of the schedule."""
        meta = plan_result.get("metadata", {})
        subjects = plan_result.get("subjects_summary", {})
        schedule = plan_result.get("schedule", {})
        warnings = plan_result.get("warnings", [])

        print("\n\n=== 🥳 YOUR CUSTOM STUDY PLAN IS READY! ===")
        print(f"Planning from {meta.get('start_day')} up to {meta.get('latest_day')}")
        
        daily_cap_info = meta.get("daily_max_hours")
        if daily_cap_info:
            print(f"Daily Study Cap: {daily_cap_info} hours (We tried hard not to exceed this!)")
        print("-------------------------------------------\n")

        print("🧠 SUBJECT BREAKDOWN:")
        for s, info in subjects.items():
            hours_info = f"Requested **{info['requested_hours']}h** | Scheduled **{info['scheduled_hours']}h**"
            print(f" - **{s}** (P{info['priority']}): {hours_info} - Deadline: {info['deadline']}")
        
        print(f"\n🗓️ DAILY SCHEDULE PREVIEW (First {number_of_days_to_show} days):")
        
        count = 0
        for day, allocation in sorted(schedule.items()):
            if count >= number_of_days_to_show:
                break
                
            total_day_load = sum(allocation.values())
            print(f"\n **{day}: Total {total_day_load:.2f} hours**")
            
            if daily_cap_info and total_day_load > daily_cap_info + 1e-9:
                print(f"   🚨 OVERLOAD: ({total_day_load:.2f}h > {daily_cap_info}h cap)")
                
            for subj, hrs in allocation.items():
                if hrs > 0:
                     print(f"    • {subj}: {hrs:.2f} hours")
                     
            count += 1
            
        if count < len(schedule):
            print(f"\n... See JSON/CSV files for the rest of the {len(schedule) - count} days.")


        if warnings:
            print("\n🚨 WARNINGS & IMPORTANT NOTES:")
            for w in warnings:
                print(f" - {w}")
            
            # Additional friendly advice
            if any("UNMET GOAL" in w for w in warnings):
                print("\nSuggestion: When goals are unmet, try adding more days to your schedule, or ask the Agent to lower the hours for lower-priority subjects!")
            elif any("Major Overload" in w for w in warnings):
                 print("\nSuggestion: That's a busy day! Consider increasing your `Daily Study Cap` or pushing a less-urgent task to a later date manually.")

# --- 🖥️ The Interactive Shell (Main User Interface) ---
def get_user_requests() -> List[Dict[str, Any]]:
    """Collects all the subject requests from the user one by one."""
    print("Let's build your to-do list! Enter each subject's details.")
    print("Date format must be: YYYY-MM-DD (e.g., 2026-03-15)")
    entries: List[Dict[str, Any]] = []
    
    while True:
        print("\n--- New Subject ---")
        subj = input("Subject Name (or just press ENTER to finish list): ").strip()
        if subj == "":
            break
            
        hrs = input(f"Total study hours needed for '{subj}': ").strip()
        dline = input(f"Deadline date for '{subj}' (YYYY-MM-DD): ").strip()
        pr = input(f"Priority (1-5, 5 is highest, default=1): ").strip()
        
        try:
            total_hours = float(hrs)
            priority = int(pr) if pr else 1
            # Just checking the date format is right before we save it
            datetime.strptime(dline, "%Y-%m-%d") 
            
            entries.append({
                "subject": subj,
                "total_hours": total_hours,
                "deadline": dline,
                "priority": priority
            })
            log.info("Added task: %s (%.1f hrs)", subj, total_hours)
            
        except ValueError:
            print("❌ Invalid input detected (Hours must be a number, Priority must be an integer). Let's try that subject again.")
        except Exception as e:
            print(f"❌ Invalid input, please check the date format: {e}")
            
    return entries

def run_sample_demo():
    """Runs a hardcoded demo to show off the features."""
    log.info("Running a sample demonstration schedule...")
    today = date.today()
    
    # Example tasks (one close, one medium, one far)
    demo_requests = [
        {"subject": "Math Finals", "total_hours": 15, "deadline": str(today + timedelta(days=7)), "priority": 3},
        {"subject": "Python Project", "total_hours": 20, "deadline": str(today + timedelta(days=5)), "priority": 5},
        {"subject": "Ancient History Reading", "total_hours": 6, "deadline": str(today + timedelta(days=14)), "priority": 1},
    ]
    
    # Planner with a tight daily cap
    planner = StudyPlannerAgent(daily_study_limit=5.5, plan_starts_on=today)
    result = planner.create_plan(demo_requests)
    
    assistant = HumanAssistant()
    assistant.show_me_the_plan(result, number_of_days_to_show=7)
    
    keeper = ScheduleFileKeeper()
    keeper.save_as_json("demo_study_plan", result)
    keeper.save_as_csv("demo_study_plan", result["day_rows_for_csv"])
    
    print("\nDemo files saved in the ./schedules/ folder.")

# --- 🚀 Main Application Startup ---
def main_application():
    """The main control flow for the user."""
    print("\n\n=== Welcome to the Study Buddy Agent! ===")
    
    while True:
        print("\nWhat do you want to do?")
        print(" 1) Give me the tasks interactively")
        print(" 2) Run the built-in demo schedule")
        print(" 3) Quit")
        
        choice = input("Your choice [1/2/3]: ").strip()
        
        if choice == "3":
            print("Goodbye! Go get some rest.")
            return
        
        if choice == "2":
            run_sample_demo()
            return
            
        if choice == "1":
            user_tasks = get_user_requests()
            if not user_tasks:
                print("No tasks entered. Try again or quit.")
                continue

            # Get general rules from the user
            cap_input = input("\nOptional: Maximum study hours you want per day (e.g., 6.0) or ENTER to skip: ").strip()
            start_input = input("Optional: Schedule Start Date (YYYY-MM-DD) or ENTER for today: ").strip()
            
            daily_limit = float(cap_input) if cap_input else None
            
            try:
                start_day = datetime.strptime(start_input, "%Y-%m-%d").date() if start_input else date.today()
            except Exception:
                print("Invalid start date format. Using today.")
                start_day = date.today()

            # Time to plan!
            planner = StudyPlannerAgent(daily_study_limit=daily_limit, plan_starts_on=start_day)
            final_plan = planner.create_plan(user_tasks)
            
            # Show the results
            assistant = HumanAssistant()
            assistant.show_me_the_plan(final_plan, number_of_days_to_show=10)

            # Save if the user likes it
            save_it = input("\nDo you want to save this full schedule to JSON and CSV? [y/N]: ").strip().lower()
            if save_it == "y":
                file_base_name = input("Enter a file name (e.g., 'spring_exam_schedule'): ").strip() or "my_schedule"
                keeper = ScheduleFileKeeper()
                keeper.save_as_json(file_base_name, final_plan)
                keeper.save_as_csv(file_base_name, final_plan["day_rows_for_csv"])
                print("Files saved successfully!")
            
            return
            
        else:
            print("Invalid choice. Please try 1, 2, or 3.")


if __name__ == "__main__":
    try:
        main_application()
    except Exception as e:
        log.critical("A major error occurred: %s", e)
        print("\nFATAL ERROR. Please check the log for details.")
