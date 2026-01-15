def format_ids(ids):
    return ",".join(map(str, ids))

def get_summary_stats_query(proc_ids):
    procs = format_ids(proc_ids)
    return f"""
    WITH cohort_patients AS (
        SELECT DISTINCT p.person_id
        FROM "cvts_cdm"."procedure_occurrence" p
        WHERE p.procedure_concept_id IN ({procs})
    ),
    stats_cohort AS (
        SELECT 
            COUNT(DISTINCT p.person_id) as total_patients,
            COUNT(DISTINCT CASE WHEN p.gender_concept_id = 8507 THEN p.person_id END) as male,
            COUNT(DISTINCT CASE WHEN p.gender_concept_id = 8532 THEN p.person_id END) as female,
            (SELECT COUNT(DISTINCT v.visit_occurrence_id) FROM "cvts_cdm"."visit_occurrence" v WHERE v.person_id IN (SELECT person_id FROM cohort_patients)) as total_visits,
            (SELECT COUNT(DISTINCT procedure_concept_id) FROM "cvts_cdm"."procedure_occurrence" WHERE procedure_concept_id IN ({procs})) as total_proc_types,
            (SELECT COUNT(DISTINCT v.visit_occurrence_id) FROM "cvts_cdm"."visit_occurrence" v WHERE v.visit_concept_id = 9201 AND v.person_id IN (SELECT person_id FROM cohort_patients)) as total_admissions,
            (SELECT COUNT(DISTINCT d.person_id) FROM "cvts_cdm"."death" d WHERE d.person_id IN (SELECT person_id FROM cohort_patients)) as deceased
        FROM "cvts_cdm"."person" p WHERE p.person_id IN (SELECT person_id FROM cohort_patients)
    ),
    stats_all AS (
        SELECT 
            COUNT(DISTINCT p.person_id) as total_patients,
            COUNT(DISTINCT CASE WHEN p.gender_concept_id = 8507 THEN p.person_id END) as male,
            COUNT(DISTINCT CASE WHEN p.gender_concept_id = 8532 THEN p.person_id END) as female,
            (SELECT COUNT(DISTINCT visit_occurrence_id) FROM "cvts_cdm"."visit_occurrence") as total_visits,
            (SELECT COUNT(DISTINCT procedure_concept_id) FROM "cvts_cdm"."procedure_occurrence") as total_proc_types,
            (SELECT COUNT(DISTINCT visit_occurrence_id) FROM "cvts_cdm"."visit_occurrence" WHERE visit_concept_id = 9201) as total_admissions,
            (SELECT COUNT(DISTINCT person_id) FROM "cvts_cdm"."death") as deceased
        FROM "cvts_cdm"."person" p
    )
    SELECT 'Total Patients' AS metric, (SELECT total_patients FROM stats_cohort) AS "Analysis Cohort", (SELECT total_patients FROM stats_all) AS "Value(all)"
    UNION ALL SELECT 'Male', (SELECT male FROM stats_cohort), (SELECT male FROM stats_all)
    UNION ALL SELECT 'Female', (SELECT female FROM stats_cohort), (SELECT female FROM stats_all)
    UNION ALL SELECT 'Total Visits', (SELECT total_visits FROM stats_cohort), (SELECT total_visits FROM stats_all)
    UNION ALL SELECT 'Total Procedure Types Analyzed', (SELECT total_proc_types FROM stats_cohort), (SELECT total_proc_types FROM stats_all)
    UNION ALL SELECT 'Total Admission', (SELECT total_admissions FROM stats_cohort), (SELECT total_admissions FROM stats_all)
    UNION ALL SELECT 'Deceased', (SELECT deceased FROM stats_cohort), (SELECT deceased FROM stats_all);
    """

def get_procedure_stats_query(proc_ids):
    procs = format_ids(proc_ids)
    return f"""
    SELECT 
        CASE 
            WHEN procedure_concept_id = 4142628 THEN 'Aortocoronary Artery Bypass Graft'
            WHEN procedure_concept_id = 4284104 THEN 'Atrioseptoplasty'
            WHEN procedure_concept_id = 4304688 THEN 'Correction of Ventricular Septal Defect'
            WHEN procedure_concept_id = 4012932 THEN 'Double Valve Replacement'
            WHEN procedure_concept_id = 4302815 THEN 'Pericardiectomy'
            WHEN procedure_concept_id = 4019929 THEN 'Repair of Tetralogy of Fallot'
            WHEN procedure_concept_id = 4095407 THEN 'Replacement of Aortic Valve'
            WHEN procedure_concept_id = 4203153 THEN 'Replacement of Mitral Valve'
            WHEN procedure_concept_id = 4294387 THEN 'Robot Assisted Laparoscopic Coronary Artery Bypass'
            ELSE 'Other'
        END AS "Procedure Name",
        COUNT(DISTINCT person_id) AS "Distinct Patient Count",
        COUNT(DISTINCT procedure_occurrence_id) AS "Distinct Procedure Count"
    FROM "cvts_cdm"."procedure_occurrence"
    WHERE procedure_concept_id IN ({procs})
    GROUP BY procedure_concept_id
    ORDER BY "Procedure Name";
    """

def get_lab_freq_stats_query(proc_ids):
    procs = format_ids(proc_ids)
    return f"""
    SELECT 
        CASE 
            WHEN po.procedure_concept_id = 4142628 THEN 'Aortocoronary artery bypass graft'
            WHEN po.procedure_concept_id = 4284104 THEN 'Atrioseptoplasty'
            WHEN po.procedure_concept_id = 4304688 THEN 'Correction of Ventricular Septal Defect'
            WHEN po.procedure_concept_id = 4012932 THEN 'Double Valve Replacement'
            WHEN po.procedure_concept_id = 4302815 THEN 'Pericardiectomy'
            WHEN po.procedure_concept_id = 4019929 THEN 'Repair of Tetralogy of Fallot'
            WHEN po.procedure_concept_id = 4095407 THEN 'Replacement of Aortic Valve'
            WHEN po.procedure_concept_id = 4203153 THEN 'Replacement of Mitral Valve'
            WHEN po.procedure_concept_id = 4294387 THEN 'Robot Assisted Laparoscopic Coronary Artery Bypass'
            ELSE 'Other'
        END AS "Procedure",
        SUM(CASE WHEN c.concept_name ILIKE '%%Prothrombin time%%' THEN 1 ELSE 0 END) AS "Prothrombin time (PT)",
        SUM(CASE WHEN c.concept_name ILIKE '%%Hemoglobin%%' AND c.concept_name NOT ILIKE '%%Glycosylated%%' THEN 1 ELSE 0 END) AS "Hemoglobin",
        SUM(CASE WHEN c.concept_name ILIKE '%%Platelet%%' THEN 1 ELSE 0 END) AS "Platelets panel",
        SUM(CASE WHEN c.concept_name ILIKE '%%Creatinine%%' THEN 1 ELSE 0 END) AS "Creatinine Ratio",
        SUM(CASE WHEN c.concept_name ILIKE '%%Potassium%%' THEN 1 ELSE 0 END) AS "Potassium",
        SUM(CASE WHEN c.concept_name ILIKE '%%Sodium%%' THEN 1 ELSE 0 END) AS "Sodium",
        SUM(CASE WHEN c.concept_name ILIKE '%%Urea%%' AND c.concept_name ILIKE '%%Urine%%' THEN 1 ELSE 0 END) AS "Urea Urine",
        SUM(CASE WHEN c.concept_name ILIKE '%%Urea nitrogen%%' THEN 1 ELSE 0 END) AS "Urea Nitrogen",
        SUM(CASE WHEN c.concept_name ILIKE '%%Lipid%%' THEN 1 ELSE 0 END) AS "Lipid panel",
        SUM(CASE WHEN c.concept_name ILIKE '%%Glucose%%' THEN 1 ELSE 0 END) AS "Glucose IV",
        SUM(CASE WHEN c.concept_name ILIKE '%%Lactate dehydrogenase%%' AND c.concept_name NOT ILIKE '%%Urine%%' THEN 1 ELSE 0 END) AS "LDH Panel",
        SUM(CASE WHEN c.concept_name ILIKE '%%Glycosylated hemoglobin%%' OR c.concept_name ILIKE '%%HbA1c%%' THEN 1 ELSE 0 END) AS "Glycosylated Haemoglobin",
        SUM(CASE WHEN c.concept_name ILIKE '%%aPTT%%' OR c.concept_name ILIKE '%%partial thromboplastin%%' THEN 1 ELSE 0 END) AS "aPTT Ratio",
        SUM(CASE WHEN c.concept_name ILIKE '%%Lactate dehydrogenase%%' AND c.concept_name ILIKE '%%Urine%%' THEN 1 ELSE 0 END) AS "LDH Urine"
    FROM "cvts_cdm"."procedure_occurrence" po
    JOIN "cvts_cdm"."visit_occurrence" vo ON po.visit_occurrence_id = vo.visit_occurrence_id
    JOIN "cvts_cdm"."measurement" m ON m.visit_occurrence_id = vo.visit_occurrence_id
    JOIN "cvts_cdm"."concept" c ON m.measurement_concept_id = c.concept_id
    WHERE po.procedure_concept_id IN ({procs})
    GROUP BY po.procedure_concept_id
    ORDER BY "Procedure";
    """

def get_mortality_stats_query(proc_ids):
    procs = format_ids(proc_ids)
    return f"""
    WITH cohort_status AS (
        SELECT DISTINCT p.person_id,
            CASE 
                WHEN p.gender_concept_id = 8507 THEN 'Male'
                WHEN p.gender_concept_id = 8532 THEN 'Female'
                ELSE 'Other' 
            END AS gender_label,
            CASE WHEN d.death_date IS NOT NULL THEN 1 ELSE 0 END AS is_deceased
        FROM "cvts_cdm"."procedure_occurrence" po
        JOIN "cvts_cdm"."person" p ON po.person_id = p.person_id
        LEFT JOIN "cvts_cdm"."death" d ON p.person_id = d.person_id
        WHERE po.procedure_concept_id IN ({procs})
    )
    SELECT 'All' AS "Gender", COUNT(*) AS "Total Patients", SUM(is_deceased) AS "Deaths", ROUND((SUM(is_deceased) * 100.0 / COUNT(*)), 2) AS "Mortality (%%)"
    FROM cohort_status
    UNION ALL
    SELECT gender_label AS "Gender", COUNT(*) AS "Total Patients", SUM(is_deceased) AS "Deaths", ROUND((SUM(is_deceased) * 100.0 / COUNT(*)), 2) AS "Mortality (%%)"
    FROM cohort_status WHERE gender_label IN ('Male', 'Female')
    GROUP BY gender_label
    ORDER BY "Total Patients" DESC;
    """

def get_lab_summary_query(proc_ids):
    procs = format_ids(proc_ids)
    return f"""
    WITH cohort_measurements AS (
        SELECT m.person_id, m.measurement_id, c.concept_name,
            CASE 
                WHEN c.concept_name ILIKE '%%Platelet%%' THEN 'Platelets panel - Blood'
                WHEN c.concept_name ILIKE '%%Hemoglobin%%' AND c.concept_name NOT ILIKE '%%Glycosylated%%' THEN 'Hemoglobin [Presence] in Blood'
                WHEN c.concept_name ILIKE '%%Creatinine%%' THEN 'Creatinine in Peritoneal Fluid / Serum'
                WHEN c.concept_name ILIKE '%%Potassium%%' THEN 'Potassium [Mass/volume]'
                WHEN c.concept_name ILIKE '%%Sodium%%' THEN 'Sodium [Mass/mass]'
                WHEN c.concept_name ILIKE '%%Prothrombin time%%' THEN 'Prothrombin Time (PT)'
                WHEN c.concept_name ILIKE '%%Glucose%%' THEN 'Glucose IV (dose mass)'
                WHEN c.concept_name ILIKE '%%Urea%%' AND c.concept_name ILIKE '%%Urine%%' THEN 'Urea [Mass/volume] in Urine'
                WHEN c.concept_name ILIKE '%%Lipid%%' THEN 'Lipid panel - Serum or Plasma'
                WHEN c.concept_name ILIKE '%%Urea nitrogen%%' THEN 'Urea nitrogen [Moles/volume] in Blood'
                WHEN c.concept_name ILIKE '%%Lactate dehydrogenase%%' AND c.concept_name NOT ILIKE '%%Urine%%' THEN 'Lactate dehydrogenase panel'
                WHEN c.concept_name ILIKE '%%Glycosylated hemoglobin%%' OR c.concept_name ILIKE '%%HbA1c%%' THEN 'Glycosylated hemoglobin'
                WHEN c.concept_name ILIKE '%%aPTT%%' OR c.concept_name ILIKE '%%partial thromboplastin%%' THEN 'aPTT ratio'
                WHEN c.concept_name ILIKE '%%Lactate dehydrogenase%%' AND c.concept_name ILIKE '%%Urine%%' THEN 'Lactate Dehydrogenase – urine'
                ELSE NULL 
            END AS lab_category
        FROM "cvts_cdm"."measurement" m
        JOIN "cvts_cdm"."concept" c ON m.measurement_concept_id = c.concept_id
        WHERE m.person_id IN (
            SELECT DISTINCT p.person_id FROM "cvts_cdm"."procedure_occurrence" p
            WHERE p.procedure_concept_id IN ({procs})
        )
    )
    SELECT lab_category AS "Lab marker / Panel", COUNT(DISTINCT person_id) AS "Distinct Patient Count", COUNT(DISTINCT measurement_id) AS "Distinct Measurement Count"
    FROM cohort_measurements WHERE lab_category IS NOT NULL
    GROUP BY lab_category ORDER BY "Distinct Patient Count" DESC;
    """

def get_master_data_query(proc_ids):
    procs = format_ids(proc_ids)
    return f"""
    WITH inpatient_visits AS (
        SELECT v.person_id, v.visit_occurrence_id, v.visit_start_date, v.visit_end_date,
            LEAD(v.visit_start_date) OVER (PARTITION BY v.person_id ORDER BY v.visit_start_date) AS next_visit_start_date,
            LEAD(v.visit_occurrence_id) OVER (PARTITION BY v.person_id ORDER BY v.visit_start_date) AS next_visit_occurrence_id
        FROM "cvts_cdm".visit_occurrence v WHERE v.visit_concept_id = 9201
    )
    SELECT
        iv.person_id,
        CASE WHEN per.gender_concept_id = 8532 THEN 0 WHEN per.gender_concept_id = 8507 THEN 1 ELSE NULL END AS gender_code,
        CASE WHEN per.gender_concept_id = 8532 THEN 'Female' WHEN per.gender_concept_id = 8507 THEN 'Male' ELSE 'Other' END AS gender,
        EXTRACT(YEAR FROM age(p.procedure_date, MAKE_DATE(per.year_of_birth, COALESCE(per.month_of_birth, 1), COALESCE(per.day_of_birth, 1)))) AS age_at_procedure,
        p.procedure_concept_id,
        CASE 
            WHEN p.procedure_concept_id = 4142628 THEN 'Aortocoronary Artery Bypass Graft'
            WHEN p.procedure_concept_id = 4284104 THEN 'Atrioseptoplasty'
            WHEN p.procedure_concept_id = 4304688 THEN 'Correction of Ventricular Septal Defect'
            WHEN p.procedure_concept_id = 4012932 THEN 'Double Valve Replacement'
            WHEN p.procedure_concept_id = 4302815 THEN 'Pericardiectomy'
            WHEN p.procedure_concept_id = 4019929 THEN 'Repair of Tetralogy of Fallot'
            WHEN p.procedure_concept_id = 4095407 THEN 'Replacement of Aortic Valve'
            WHEN p.procedure_concept_id = 4203153 THEN 'Replacement of Mitral Valve'
            WHEN p.procedure_concept_id = 4294387 THEN 'Robot Assisted Laparoscopic Coronary Artery Bypass'
            ELSE 'Other'
        END AS procedure_name,
        iv.visit_start_date AS index_admit_date,
        p.procedure_date,
        iv.next_visit_start_date AS readmission_date,
        (iv.next_visit_start_date - iv.visit_end_date) AS gap_days,
        CASE WHEN d.person_id IS NOT NULL THEN 1 ELSE 0 END AS is_death,
        CASE WHEN d.death_date IS NOT NULL THEN (d.death_date - p.procedure_date) ELSE NULL END AS days_proc_to_death,
        (iv.visit_end_date - iv.visit_start_date) AS length_of_stay,
        CASE WHEN (iv.visit_end_date - p.procedure_date) < 0 THEN 0 ELSE (iv.visit_end_date - p.procedure_date) END AS post_procedure_visit_los,
        CASE WHEN (iv.next_visit_start_date IS NOT NULL AND iv.next_visit_start_date <= iv.visit_end_date + INTERVAL '30 day') OR ((iv.visit_end_date - p.procedure_date) > 10) THEN 1 ELSE 0 END AS is_complicated
    FROM inpatient_visits iv
    JOIN "cvts_cdm".procedure_occurrence p ON iv.visit_occurrence_id = p.visit_occurrence_id
    JOIN "cvts_cdm".person per ON iv.person_id = per.person_id
    LEFT JOIN "cvts_cdm".death d ON iv.person_id = d.person_id
    WHERE p.procedure_concept_id IN ({procs})
    ORDER BY iv.person_id, iv.visit_start_date;
    """

def get_lab_measurement_query(proc_ids, meas_ids):
    procs = format_ids(proc_ids)
    meas = format_ids(meas_ids)
    return f"""
    SELECT 
        p.procedure_concept_id, m.measurement_concept_id, m.person_id,
        m.value_as_number as labmarker_value, m.measurement_datetime,
        p.procedure_date as procedure_datetime,
        (v.visit_end_date - v.visit_start_date) as los_days,
        m.range_low, m.range_high
    FROM "cvts_cdm".measurement m
    JOIN "cvts_cdm".visit_occurrence v ON m.visit_occurrence_id = v.visit_occurrence_id
    JOIN "cvts_cdm".procedure_occurrence p ON v.visit_occurrence_id = p.visit_occurrence_id
    WHERE p.procedure_concept_id IN ({procs})
      AND m.measurement_concept_id IN ({meas})
      AND m.value_as_number IS NOT NULL;
    """