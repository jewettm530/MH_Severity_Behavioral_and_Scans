# The Transdiagnostic Connectome Project

A richly phenotyped transdiagnostic dataset with behavioral and Magnetic Resonance Imaging (MRI) data from 241 individuals aged 18 to 70, comprising 148 individuals meeting diagnostic criteria for a broad range of psychiatric illnesses and a healthy comparison group of 93 individuals.

These data include high-resolution anatomical scans and 6 x resting-state, and 3 x task-based (2 x Stroop, 1 x Faces/Shapes) functional MRI runs. Participants completed over 50 psychological and cognitive questionnaires, as well as a semi-structured clinical interview.

Data was collected at the Brain Imaging Center, Yale University, New Haven, CT and McLean Hospital, Belmont, MA. This dataset will allow investigation into brain function and transdiagnostic psychopathology in a community sample. See preprint (https://www.medrxiv.org/content/10.1101/2024.06.18.24309054v1) and below for detailed information. 

### Inclusion Criteria

Participants in the study met the following inclusion criteria:

* Aged 18 to 64 years and spoke English
* No metal contraindications, no history of concussion or prior neurological problems, no color-blindness
* Prior history of affective or psychotic illness or no psychiatric history

### Exclusion criteria

Participants meeting any of the criteria listed below were excluded from the study:
* Neurological disorders
* Pervasive developmental disorders (e.g., autism spectrum disorder)
* Any medical condition that increases risk for MRI (e.g., pacemaker, dental braces)
* MRI contraindications (e.g., claustrophobia pregnancy)

### Consent

Institutional Review Board approval and consent were obtained. To characterise the sample, we collected data on race/ethnicity, income, use of psychotropic medication, and family history of medical or psychiatric conditions.

## Clinical Measures

### Completed by Participants:
* Health and demographics questionnaire
* Alcohol Tobacco Caffeine Use Questionnaire (ATC)
* Broad Autism Phenotype Questionnaire (BAPQ-2)
* Barratt Impulsiveness Scale (BIS)
* Behavioral Inhibition/Activation Scale (BISBAS)
* Childhood Trauma Questionnaire (CTQ)
* Domain Specific Risk Taking (DOSPERT)
* Fagerstrom Test for Nicotine Dependence (FTND)
* NEO Five Factor Inventory (NEO-FFI)
* Quick Inventory of Depressive Symptomatology (QIDS)
* Multidimensional Scale for Perceived Social Support (MSPSS)
* State-Trait Anxiety Inventory (STAI)
* Temperament Character Inventory (TCI)
* Anxiety Sensitivity Index (ASI)
* Depression Anxiety Stress Scale (DASS)
* Profile of Mood States (POMS)
* Perceived Stress Scale (PSS)
* Shipley Institute of Living Scale (Shipley)
* Temporal Experience of Pleasure Scale (TEPS)
* Cognitive Emotion Regulation Questionnaire (CERQ)
* Cognitive Failures Questionnaire (CFQ)
* Cognitive Reflections Test (CRT)
* Experiences in Close Relationships Inventory (ECR)
* Positive Urgency Measure (PUM) 
* Ruminative Responses Scale (RRS)
* Retrospective Self-Report of Inhibition (RSRI)
* Snaith-Hamilton Pleasure Scale (SHAPS)
* Test My Brain (TMB)
* Stroop Task (during fMRI)
* Hammer Task (during fMRI)

### Completed by Clinicians:
* Structured Clinical Interview for DSM-5 Disorder (SCID-5)
* Clinical Global Impression (CGI)
* Anxiety Symptom Chronicity (ASC)
* Columbia Suicide Severity Rating Scale (CSSR-S)
* Range of Impaired Functioning Tool (LIFE-RIFT)
* Montgomery-Asberg Depression Rating Scale (MADRS)
* Multnomah Community Ability Scale (MCAS)
* Positive and Negative Syndrome Scale (PANSS)
* Panic Disorder Severity Scale (PDSS)
* Young Mania Rating Scale (YMRS)

### Clinical Measures Data

Relevant clinical measures can be found in the `phenotype` folder, with each measure and its items described in the relevant `_definition` .csv file. The 'qc' columns indicate quality control checks done on each measure (i.e.,  number of unanswered items by a participant.) '999' values indicate missing or skipped data.

## MRI acquisition parameters
MRI data were acquired at both sites using harmonized Siemens Magnetom 3T Prisma MRI scanners and a 64-channel head coil. T1-weighted (T1-w) anatomical images were acquired using a multi-echo MPRAGE sequence following parameters: acquisition duration of 132 seconds, with a repetition time (TR) of 2.2 seconds, echo times (TE) of 1.5, 3.4, 5.2, and 7.0 milliseconds, a flip angle of 7°, an inversion time (TI) of 1.1 seconds, a sagittal orientation and anterior (A) to posterior (P) phase encoding. The slice thickness was 1.2 millimeters, and 144 slices were acquired. The image resolution was 1.2 mm3. A root mean square of the four images corresponding to each echo was computed to derive a single image. T2-weighted (T2w) anatomical images with the following parameters: TR of 2800 milliseconds, TE of 326 milliseconds, a sagittal orientation, and AP phase encoding direction. The slice thickness was 1.2 millimeters, and 144 slices were acquired. All seven functional MRI runs were acquired with the same parameters matching the HCP protocol6,9, varying only the conditions (rest/task) and separately acquired phase encoding directions (AP/PA). For the resting-state, Stroop task, and Emotional Faces task, a total of 488, 510, and 493 volumes were acquired, respectively, all using the following MRI sequence parameters: TR = 800 milliseconds, TE = 37 milliseconds, flip angle = 52°, and voxel size =2mm3. A multi-band acceleration factor of 8 was applied. An auto-align pulse sequence protocol was used to align the acquisition slices of the functional scans parallel to the anterior. To enable the correction of the distortions in the EPI images, B0-field maps were acquired in both AP and PA directions with a standard Spin Echo sequence. Detailed MRI acquisition protocols for both sites are available in Appendix B. In total, four resting-state (2 AP, 2 PA), 2 Stroop task acquisitions (1 AP [Block 1], 1 PA [Block 2]), and 1 Emotional Faces task acquisition (1 AP) acquisitions were collected. Select participants out of the total sample did not complete each functional neuroimaging run; thus the sample sizes for each run were as follows: resting-state AP run 1, n = 241; resting-state PA run 1, n = 241; resting-state AP run 2, n = 237; resting-state AP run 2, n = 235; Stroop task AP, n = 226; Stroop task PA, n = 224; and Emotional Faces task AP, n = 226. 

For the Emotional Faces task, the faces are fear and anger expressing (male and female groups) from the NimStim database. The faces used in each trial are outlines in each events.tsv file.For example, FA1 = female anger stimuli set number 1, or FF1 =female fear stimuli set number 1. Unfortunately, we cannot release the actual images publicly. An important consideration here might be that this task has no neutral control nor positively valenced comparison for faces (i.e., is precisely a negatively valenced face vs non-face/shape version of the task). We will soon update the events.tsv files on OpenNeuro with more informative file names (e.g. female_fear, female_anger, male_fear, male_anger). 


Detailed information and protocols regarding the dataset can be found here: https://www.medrxiv.org/content/10.1101/2024.06.18.24309054v1




# Reduced-feature TCP Random Forest pipeline

Use these files to replace the matching files in your `scripts/` folder.

## What changed

Script 02 now reduces behavioral features before modeling:

- Loads all non-definition phenotype CSV files.
- Uses `qids01.csv` only for the target `qvtot`.
- Removes timing/click/page-submit columns.
- Removes QC/nonresponse/count metadata.
- Removes notes/comments/free-text fields.
- Removes item-level questionnaire columns like `bapq_q1`, `bapq_q2`, etc.
- Keeps likely total scores, subscale scores, averages, psychiatric trait scores, and cognitive performance summaries.
- Drops columns with 25% or more missing values.

## Run order

You do not need to rerun script 03 unless you changed the imaging extraction.

```bash
python3 scripts/02_prepare_behavioral_data.py
python3 scripts/04_merge_features.py
python3 scripts/05_train_random_forest.py
python3 scripts/06_evaluate_results.py
```

## About the “automatic feature selection” idea

I did not add SelectKBest/PCA/Lasso selection here because it adds another layer that is harder to explain. Instead, this version uses interpretable rule-based feature reduction before modeling.
