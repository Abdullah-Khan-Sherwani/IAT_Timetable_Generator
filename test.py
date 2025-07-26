from dataclasses import dataclass
from typing import List
import itertools

DAYS = ["Mon", "Tue", "Wed", "Thu"]
START_TIMES = ["08:30", "09:45", "11:00", "12:15", "13:30", "14:45", "16:00"]

def time_to_minutes(time: str) -> int:
    h, m = map(int, time.split(":"))
    return h * 60 + m

def minutes_to_time(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"

@dataclass
class TimeSlot:
    day: str
    start_time: str
    duration: int  # in minutes

    def end_time(self) -> str:
        return minutes_to_time(time_to_minutes(self.start_time) + self.duration)

    def overlaps(self, other: 'TimeSlot') -> bool:
        if self.day != other.day:
            return False
        start1 = time_to_minutes(self.start_time)
        end1 = start1 + self.duration
        start2 = time_to_minutes(other.start_time)
        end2 = start2 + other.duration
        return max(start1, start2) < min(end1, end2)

@dataclass
class CourseOption:
    time_slots: List[TimeSlot]
    has_lab: bool
    teacher: str
    priority: int

@dataclass
class Course:
    name: str
    options: List[CourseOption]

def is_valid_schedule(course_options: List[CourseOption]) -> bool:
    all_slots = list(itertools.chain.from_iterable(opt.time_slots for opt in course_options))

    # Check time overlaps
    for i in range(len(all_slots)):
        for j in range(i + 1, len(all_slots)):
            if all_slots[i].overlaps(all_slots[j]):
                return False

    # Ensure labs (if any) are on a single day
    for opt in course_options:
        if opt.has_lab:
            lab_days = [slot.day for slot in opt.time_slots if slot.duration > 90]
            if len(set(lab_days)) != 1:
                return False

    return True

def compute_score(course_options: List[CourseOption]) -> int:
    return sum(opt.priority for opt in course_options)

def create_courses() -> List[Course]:
    one_hr_class = 75
    lab_duration = 165

    software_eng = Course("Software Engineering", [
        CourseOption(
            time_slots=[TimeSlot("Mon", "08:30", one_hr_class), TimeSlot("Wed", "08:30", one_hr_class)],
            has_lab=False, teacher="Mr Ahsan Jawed", priority=2),
        CourseOption(
            time_slots=[TimeSlot("Mon", "11:30", one_hr_class), TimeSlot("Wed", "11:30", one_hr_class)],
            has_lab=False, teacher="TBA", priority=2),
        CourseOption(
            time_slots=[TimeSlot("Tue", "11:30", one_hr_class), TimeSlot("Thu", "11:30", one_hr_class)],
            has_lab=False, teacher="TBA", priority=2),
    ])

    operating_sys = Course("Operating Systems", [
        CourseOption(
            time_slots=[TimeSlot("Mon", "08:30", one_hr_class), TimeSlot("Wed", "08:30", one_hr_class)],
            has_lab=False, teacher="Sir Waseem", priority=0),
        CourseOption(
            time_slots=[TimeSlot("Mon", "10:00", one_hr_class), TimeSlot("Wed", "10:00", one_hr_class)],
            has_lab=False, teacher="Sir Waseem", priority=0),
        CourseOption(
            time_slots=[TimeSlot("Tue", "10:00", one_hr_class), TimeSlot("Thu", "10:00", one_hr_class)],
            has_lab=False, teacher="Sir Salman Zaffar", priority=1),
        CourseOption(
            time_slots=[TimeSlot("Tue", "11:30", one_hr_class), TimeSlot("Thu", "11:30", one_hr_class)],
            has_lab=False, teacher="Sir Salman Zaffar", priority=1),
    ])

    database_sys = Course("Database Systems", [
        CourseOption(
            time_slots=[
                TimeSlot("Mon", "11:30", one_hr_class),
                TimeSlot("Wed", "11:30", one_hr_class),
                TimeSlot("Thu", "11:30", lab_duration)],
            has_lab=True, teacher="Ms Abeera Tariq", priority=2),
        CourseOption(
            time_slots=[
                TimeSlot("Mon", "10:00", one_hr_class),
                TimeSlot("Wed", "10:00", one_hr_class),
                TimeSlot("Thu", "08:30", lab_duration)],
            has_lab=True, teacher="Ms Abeera Tariq", priority=2),
    ])

    business_comm = Course("Business Communication", [
        CourseOption(
            time_slots=[TimeSlot("Tue", "11:30", one_hr_class), TimeSlot("Thu", "11:30", one_hr_class)],
            has_lab=False, teacher="Ms. Talat Davis", priority=2),
        CourseOption(
            time_slots=[TimeSlot("Tue", "13:00", one_hr_class), TimeSlot("Thu", "13:00", one_hr_class)],
            has_lab=False, teacher="Ms. Talat Davis", priority=2),
    ])

    elective = Course("CS Elective", [
        CourseOption(
            time_slots=[TimeSlot("Mon", "11:30", one_hr_class), TimeSlot("Wed", "11:30", one_hr_class)],
            has_lab=False, teacher="Business Intelligence by TBA", priority=1),
        CourseOption(
            time_slots=[TimeSlot("Mon", "13:00", one_hr_class), TimeSlot("Wed", "13:00", one_hr_class)],
            has_lab=False, teacher="Data Warehousing by Dr Tariq Mehmood", priority=1),
        CourseOption(
            time_slots=[TimeSlot("Tue", "13:00", one_hr_class), TimeSlot("Thu", "13:00", one_hr_class)],
            has_lab=False, teacher="Web Based App Dev by Mr Adil Saleem", priority=2),
        CourseOption(
            time_slots=[TimeSlot("Tue", "14:30", one_hr_class), TimeSlot("Thu", "14:30", one_hr_class)],
            has_lab=False, teacher="Computer Security by Dr Faisal Iradat", priority=2),
        # CourseOption(
        #     time_slots=[TimeSlot("Tue", "8:30", one_hr_class), TimeSlot("Thu", "8:30", one_hr_class)],
        #     has_lab=False, teacher="Introduction to Machine Learning", priority=3),
    ])

    return [software_eng, operating_sys, database_sys, business_comm, elective]

def find_all_valid_schedules():
    courses = create_courses()
    all_valid_schedules = []

    for combo in itertools.product(*(course.options for course in courses)):
        if is_valid_schedule(combo):
            score = compute_score(combo)
            all_valid_schedules.append((combo, score))

    # Sort by score descending
    all_valid_schedules.sort(key=lambda x: x[1], reverse=True)
    return all_valid_schedules

# Run CSP Solver
if __name__ == "__main__":
    valid_schedules = find_all_valid_schedules()

    if valid_schedules:
        for i, (schedule, score) in enumerate(valid_schedules, start=1):
            print(f"\n🔹 Valid Schedule #{i} — Total Priority Score: {score}")
            for course, option in zip(create_courses(), schedule):
                print(f"\n📘 {course.name} — Teacher: {option.teacher} (Priority: {option.priority})")
                for ts in option.time_slots:
                    label = "Lab" if ts.duration > 90 else "Class"
                    print(f"  📅 {ts.day} {ts.start_time}–{ts.end_time()} ({label})")
    else:
        print("❌ No valid schedules found.")
