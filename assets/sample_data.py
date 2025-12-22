"""
Sample Airline Operations Data for Demonstration
Simulates SOP documents, maintenance logs, and operational procedures
"""

# Sample Standard Operating Procedures
SAMPLE_SOPS = [
    {
        "id": "SOP-B737-001",
        "title": "B737 Pre-Flight Inspection Checklist",
        "fleet_type": "B737",
        "document_type": "sop",
        "content": """
PRE-FLIGHT INSPECTION PROCEDURE - BOEING 737

CHAPTER 1: EXTERIOR INSPECTION

1.1 NOSE SECTION
- Check nose gear tire condition and pressure (185 PSI nominal)
- Verify pitot tubes are clear and undamaged
- Inspect windshield for cracks or damage
- Confirm all static ports are unobstructed

1.2 LEFT WING
- Inspect leading edge slats for damage
- Check fuel tank vents are clear
- Verify navigation lights operational
- Examine flap tracks and fairings

1.3 ENGINE #1 (LEFT)
- Ensure all cowling latches secure
- Check fan blades for FOD damage
- Verify no fluid leaks (oil, fuel, hydraulic)
- Inspect thrust reverser blocker doors

WARNING: Never approach operating engine within 25 feet of inlet.

1.4 FUSELAGE LEFT SIDE
- Check all doors and hatches secure
- Verify emergency exits not blocked
- Inspect antennas and sensors

CHAPTER 2: COCKPIT PREPARATION

2.1 INITIAL SETUP
Step 1: Verify parking brake SET
Step 2: Confirm battery switch ON
Step 3: Check emergency lights ARMED
Step 4: Verify oxygen pressure above 1600 PSI

2.2 FLIGHT MANAGEMENT SYSTEM
Step 1: Enter flight plan from dispatch
Step 2: Verify waypoints against release
Step 3: Cross-check fuel load with dispatch
Step 4: Confirm alternate airport entered

CAUTION: Always verify FMS data against paper flight plan.
"""
    },
    {
        "id": "SOP-B787-002", 
        "title": "B787 Dreamliner Engine Start Procedure",
        "fleet_type": "B787",
        "document_type": "sop",
        "content": """
ENGINE START PROCEDURE - BOEING 787 DREAMLINER

PREREQUISITES:
- APU running or external power connected
- Hydraulic pressure normal (3000 PSI)
- Engine anti-ice OFF
- Parking brake SET

PROCEDURE:

1. ENGINE START SEQUENCE
Step 1: Ensure thrust levers at IDLE
Step 2: Set START selector to desired engine (1 or 2)
Step 3: Monitor EGT rise (maximum start limit: 1050°C)
Step 4: At N2 > 20%, move fuel lever to RUN
Step 5: Monitor oil pressure rise (minimum 25 PSI within 30 seconds)
Step 6: Verify stable idle parameters:
   - N1: 19-23%
   - N2: 58-62%
   - EGT: 400-500°C
   - Oil pressure: 65-90 PSI

2. POST-START CHECKS
- Confirm engine generator online
- Verify hydraulic systems pressurized
- Check bleed air valve positions
- Monitor for abnormal vibrations

CAUTION: If EGT exceeds 1050°C during start, abort immediately.

ABORT PROCEDURE:
Step 1: Fuel lever to CUTOFF
Step 2: START selector to OFF
Step 3: Allow 5 minutes cooling before retry
Step 4: Log event in aircraft maintenance log
"""
    },
    {
        "id": "SOP-DELAY-001",
        "title": "Delay Management Procedures",
        "document_type": "sop",
        "content": """
DELAY MANAGEMENT AND ROOT CAUSE ANALYSIS

1. DELAY CATEGORIES

1.1 MAINTENANCE DELAYS (Code 41-47)
- Code 41: Scheduled maintenance overrun
- Code 42: Unscheduled maintenance required
- Code 43: MEL/CDL item deferral
- Code 44: Specialized equipment required
- Code 45: Parts unavailable
- Code 46: Maintenance staff shortage
- Code 47: Documentation incomplete

1.2 CREW DELAYS (Code 61-67)
- Code 61: Crew late report
- Code 62: Crew duty time exceeded
- Code 63: Crew rest requirement
- Code 64: Crew training/qualification
- Code 65: Crew sickness
- Code 66: Crew reassignment required

1.3 ATC/WEATHER DELAYS (Code 71-79)
- Code 71: Ground stop
- Code 72: Flow control
- Code 73: Weather at departure
- Code 74: Weather at arrival
- Code 75: En route weather deviation
- Code 76: Airport capacity constraints

2. MITIGATION PROCEDURES

2.1 MAINTENANCE DELAYS
Step 1: Assess nature of defect
Step 2: Check MEL applicability
Step 3: Coordinate with MCC for options:
   - MEL deferral (if applicable)
   - Aircraft swap
   - Spare aircraft assignment
Step 4: Update passengers at 15-minute intervals
Step 5: Provide meal vouchers for delays > 2 hours

2.2 ESTIMATED COMPLETION TIMES
- Minor repairs: 30-60 minutes
- MEL deferral paperwork: 15-30 minutes
- Major component replacement: 2-4 hours
- Aircraft swap: 45-90 minutes
"""
    }
]

SAMPLE_MEL = [
    {
        "id": "MEL-21-31",
        "system": "Air Conditioning",
        "title": "Pack Valve Inoperative",
        "fleet_type": "B737",
        "content": """
MEL ITEM 21-31: AIR CONDITIONING PACK VALVE

ITEM: One pack valve may be inoperative.

CONDITIONS:
(M) One pack valve may be inoperative provided:
a) Flight does not exceed FL350
b) Remaining pack operates normally
c) Associated recirculation fan is operative
d) Pressurization system operates normally

OPERATIONS PROCEDURES:
1. Maximum altitude: FL350
2. Passenger cabin altitude not to exceed 8000 feet
3. Brief cabin crew on reduced system redundancy
4. Consider route that allows lower altitudes

MAINTENANCE PROCEDURES:
1. Determine which pack is affected (L or R)
2. Verify system isolation
3. Document in aircraft logbook
4. Placard associated controls

CATEGORY: C (10 days)
INTERVAL: 10 calendar days

RECTIFICATION INTERVAL:
This item must be rectified within 10 calendar days of initial deferral.
"""
    }
]

SAMPLE_MAINTENANCE_LOG = [
    {
        "id": "MAINT-2024-001",
        "flight": "UA234",
        "aircraft": "N78881",
        "fleet_type": "B787",
        "content": """
MAINTENANCE LOG ENTRY

Flight: UA234
Aircraft: N78881 (Boeing 787-8)
Date: 2024-01-15
Station: ORD

DISCREPANCY REPORTED:
- Engine 2 (RH) vibration indication high during climb
- N1 vibration: 2.8 (limit: 3.0)
- N2 vibration: 1.9 (limit: 2.5)
- No abnormal sounds reported by crew

ACTION TAKEN:
1. Performed engine ground run
2. Monitored vibration levels - within limits on ground
3. Borescope inspection of fan blades - no damage found
4. Cleaned fan blade leading edges
5. Engine run post-cleaning - vibration reduced to:
   - N1: 1.2
   - N2: 0.8

PARTS REPLACED: None
WORK TIME: 2.5 hours
DELAY: 45 minutes

RELEASE TO SERVICE:
Aircraft released to service per Boeing MM 72-00-00.
Vibration levels within normal range.
Recommend monitoring on next 5 flights.

Signed: J. Smith, AME License #12345
"""
    }
]


def get_all_sample_documents():
    """Get all sample documents for seeding the knowledge base."""
    documents = []
    
    for sop in SAMPLE_SOPS:
        documents.append({
            "chunk_id": sop["id"],
            "content": sop["content"],
            "metadata": {
                "source_file": f"{sop['id']}.pdf",
                "document_type": sop.get("document_type", "sop"),
                "fleet_type": sop.get("fleet_type"),
                "title": sop["title"]
            }
        })
    
    for mel in SAMPLE_MEL:
        documents.append({
            "chunk_id": mel["id"],
            "content": mel["content"],
            "metadata": {
                "source_file": "MEL_MASTER.pdf",
                "document_type": "mel",
                "fleet_type": mel.get("fleet_type"),
                "title": mel["title"]
            }
        })
    
    for log in SAMPLE_MAINTENANCE_LOG:
        documents.append({
            "chunk_id": log["id"],
            "content": log["content"],
            "metadata": {
                "source_file": "MAINTENANCE_LOGS.pdf",
                "document_type": "maintenance",
                "fleet_type": log.get("fleet_type"),
                "flight": log.get("flight")
            }
        })
    
    return documents

