# Packages
# Basic packages
import numpy as np

from scipy import integrate, stats, spatial
from scipy.special import expit, binom

import pandas as pd
import xlrd # help read excel files directly from source into pandas

import copy
import warnings

# Building parameter/computation graph
import inspect
from collections import OrderedDict

# OS/filesystem tools
import time
from datetime import datetime
import random
import string
import os
import shutil
import sys
import cloudpickle

# Distributed computing tools
import dask
import distributed
from dask.distributed import Client
from dask.distributed import as_completed
import itertools


# State Dimensions
# Health states (S, E and D are fixed to 1 dimension)
nI_symp = 2 # number of sympyomatic infected states
nI = 2+nI_symp # number of total infected states (disease stages), the +2 are Exposed and I_nonsymptomatic
nR = 2 # number of recovery states (antibody development post-disease, IgM and IgG are two stages)
nHS = 2+nI+nR # number of total health states, the +2: S, D are suspectible and dead

# Age groups (risk groups)
nAge = 9 # In accordance w Imperial #13 report (0-9, 10-19, ... 70-79, 80+)

# Isolation states
nIso = 4 # None/distancing, Case isolation, Hospitalised, Hospital staff

# Testing states
nTest = 4 # untested/negative, Virus positive, Antibody positive, Both positive


stateTensor = np.ones((nAge, nHS, nIso, nTest))


# Population (data from Imperial #13 ages.csv/UK)
agePopulationTotal = 1000.*np.array([8044.056,7642.473,8558.707,9295.024,8604.251,9173.465,7286.777,5830.635,3450.616])
#agePopulationTotal = 1000.*pd.read_csv("https://raw.githubusercontent.com/ImperialCollegeLondon/covid19model/master/data/ages.csv").iloc[3].values[2:]
# Currently: let's work with england population only instead of full UK, as NHS England + CHESS data is much clearer than other regions
agePopulationTotal *= 55.98/66.27 # (google england/uk population 2018, assuming age dist is similar)
agePopulationRatio = agePopulationTotal/np.sum(agePopulationTotal)

# Helper function to adjust average rates to age-aware rates
def adjustRatesByAge_KeepAverageRate(rate, ageRelativeAdjustment, agePopulationRatio=agePopulationRatio, maxOutRate=10):
    """This is a helper function and wont be picked up as a model parameter!"""
    if rate == 0:
        return np.zeros_like(ageRelativeAdjustment)
    if rate >= maxOutRate:
        warnings.warn("covidTesting::adjustRatesByAge_KeepAverageRate Input rate {} > maxOutRate {}, returning input rates".format(rate, maxOutRate))
        return rate*np.ones_like(ageRelativeAdjustment)
    out = np.zeros_like(ageRelativeAdjustment)
    out[0] = maxOutRate+1 # just to start the while loop below
    while np.sum(out>=maxOutRate)>0:
        corrFactor = np.sum(agePopulationRatio/(1+ageRelativeAdjustment))
        out =  rate * (1+ageRelativeAdjustment) * corrFactor
        if np.sum(out>=maxOutRate)>0:
            warnings.warn("covidTesting::adjustRatesByAge_KeepAverageRate Adjusted rate larger than {} encountered, reducing ageAdjustment variance by 10%".format(maxOutRate))
            tmp_mean = np.mean(ageRelativeAdjustment)
            ageRelativeAdjustment = tmp_mean + np.sqrt(0.9)*(ageRelativeAdjustment-tmp_mean)
    return out

# For calculations see data_cleaning_py.ipynb, calculations from NHS England dataset as per 05 Apr
relativeDeathRisk_given_COVID_by_age = np.array([-0.99742186, -0.99728639, -0.98158438, -0.9830432 , -0.82983414,
       -0.84039294,  0.10768979,  0.38432409,  5.13754904])

#ageRelativeDiseaseSeverity = np.array([-0.8, -0.6, -0.3, -0.3, -0.1, 0.1, 0.35, 0.4, 0.5]) # FIXED (above) - this is a guess, find data and fix
#ageRelativeRecoverySpeed = np.array([0.2]*5+[-0.1, -0.2, -0.3, -0.5]) # TODO - this is a guess, find data and fix
ageRelativeRecoverySpeed = np.array([0.]*9) # For now we make it same for everyone, makes calculations easier

# For calculations see data_cleaning_py.ipynb, calculations from NHS England dataset as per 05 Apr
caseFatalityRatioHospital_given_COVID_by_age = np.array([0.00856164, 0.03768844, 0.02321319, 0.04282494, 0.07512237,
       0.12550367, 0.167096  , 0.37953452, 0.45757006])

def trFunc_diseaseProgression(
    # Basic parameters to adhere to
    nonsymptomatic_ratio = 0.86,

    # number of days between measurable events
    infect_to_symptoms = 5.,
    #symptom_to_death = 16.,
    symptom_to_recovery = 10., # 20.5, #unrealiticly long for old people
    symptom_to_hospitalisation = 5.76,
    hospitalisation_to_recovery = 14.51,
    IgG_formation = 15.,

    # Age related parameters
    # for now we'll assume that all hospitalised cases are known (overall 23% of hospitalised COVID patients die. 9% overall case fatality ratio)
    caseFatalityRatioHospital_given_COVID_by_age = caseFatalityRatioHospital_given_COVID_by_age,
    ageRelativeRecoverySpeed = ageRelativeRecoverySpeed,

    # Unknown rates to estimate
    nonsymp_to_recovery = 15.,
    inverse_IS1_IS2 = 4.,


    **kwargs
    ):
    # Now we have all the information to build the age-aware multistage SIR model transition matrix
    # The full transition tensor is a sparse map from the Age x HealthState x isolation state to HealthState,
        # and thus is a 4th order tensor itself, representing a linear mapping
        # from "number of people aged A in health state B and isolation state C to health state D.
    trTensor_diseaseProgression = np.zeros((nAge, nHS, nIso, nHS))
    # Use basic parameters to regularise inputs
    E_IS1 = 1./infect_to_symptoms
    # Numbers nonsymptomatic is assumed to be 86% -> E->IN / E-IS1 = 0.86/0.14
    E_IN = 0.86/0.14 * E_IS1

    # Nonsymptomatic recovery
    IN_R1 = 1./nonsymp_to_recovery

    IS1_IS2  = 1./inverse_IS1_IS2

    IS2_R1 = 1./(symptom_to_recovery-inverse_IS1_IS2)

    R1_R2 = 1./IgG_formation


    # Disease progression matrix # TODO - calibrate (together with transmissionInfectionStage)
    # rows: from-state, cols: to-state (non-symmetric!)
    # - this represent excess deaths only, doesn't contain baseline deaths!

    # Calculate all non-serious cases that do not end up in hospitals.
    # Note that we only have reliable death data from hospitals (NHS England), so we do not model people dieing outside hospitals
    diseaseProgBaseline = np.array([
    # to: E,   IN,   IS1,   IS2,    R1,   R2,   D
        [  0 , E_IN, E_IS1,    0,   0,     0,   0   ], # from E
        [  0,   0,     0,   0,    IN_R1,   0,   0   ], # from IN
        [  0 ,  0,     0, IS1_IS2,  0,     0,    0 ], # from IS1
        [  0 ,  0,     0,    0,  IS2_R1,   0,   0  ], # from IS2
        [  0 ,  0,     0,    0,    0,    R1_R2,  0   ], # from R1
        [  0 ,  0,     0,    0,    0,     0,   0   ], # from R2
        [  0 ,  0,     0,    0,    0,     0,   0   ] # from D
    ])

    ageAdjusted_diseaseProgBaseline = copy.deepcopy(np.repeat(diseaseProgBaseline[np.newaxis],nAge,axis=0))
    # Modify all death and R1 rates:
    for ii in range(ageAdjusted_diseaseProgBaseline.shape[1]):
        # Adjust death rate by age dependent disease severity
        ageAdjusted_diseaseProgBaseline[:,ii,-1] = adjustRatesByAge_KeepAverageRate(
            ageAdjusted_diseaseProgBaseline[0,ii,-1],
            ageRelativeAdjustment=relativeDeathRisk_given_COVID_by_age
        )

        # Adjust recovery rate by age dependent recovery speed
        ageAdjusted_diseaseProgBaseline[:,ii,-3] = adjustRatesByAge_KeepAverageRate(
            ageAdjusted_diseaseProgBaseline[0,ii,-3],
            ageRelativeAdjustment=ageRelativeRecoverySpeed,
            agePopulationRatio=agePopulationRatio
        )

    ageAdjusted_diseaseProgBaseline_Hospital = copy.deepcopy(ageAdjusted_diseaseProgBaseline)
    # Calculate hospitalisation based rates, for which we do have data. Hospitalisation can end up with deaths
    # Make sure that the ratio of recoveries in hospital honour the case fatality ratio appropriately
    # IS2 -> death
    ageAdjusted_diseaseProgBaseline_Hospital[:,3,-1] = (
        # IS2 -> recovery
        ageAdjusted_diseaseProgBaseline_Hospital[:,3,-3] * (
            # multiply by cfr / (1-cfr) to get correct rate towards death
            caseFatalityRatioHospital_given_COVID_by_age/(
                 1 -  caseFatalityRatioHospital_given_COVID_by_age)
        )
    )


    # TODO - time to death might be incorrect overall without an extra delay state, especially for young people

    # Non-hospitalised disease progression
    for i1 in [0,1,3]:
        trTensor_diseaseProgression[:,1:,i1,1:] = ageAdjusted_diseaseProgBaseline

    # hospitalised disease progression
    trTensor_diseaseProgression[:,1:,2,1:] = ageAdjusted_diseaseProgBaseline_Hospital


    return trTensor_diseaseProgression

# Larger data driver approaches, with age distribution, see data_cleaning_R.ipynb for details
ageHospitalisationRateBaseline = pd.read_csv('../data/clean_hosp-epis-stat-admi-summ-rep-2015-16-rep_table_6.csv', sep=',').iloc[:,-1].values
ageHospitalisationRecoveryRateBaseline = 1./pd.read_csv('../data/clean_10641_LoS_age_provider_suppressed.csv', sep=',').iloc[:,-1].values

# Calculate initial hospitalisation (occupancy), that will be used to initialise the model
initBaselineHospitalOccupancyEquilibriumAgeRatio = ageHospitalisationRateBaseline/(ageHospitalisationRateBaseline+ageHospitalisationRecoveryRateBaseline)


# Take into account the NHS work-force in hospitals that for our purposes count as "hospitalised S" population,
# also unaffected by quarantine measures
ageNhsClinicalStaffPopulationRatio = pd.read_csv('../data/clean_nhsclinicalstaff.csv', sep=',').iloc[:,-1].values

# Extra rate of hospitalisation due to COVID-19 infection stages
# TODO - find / estimate data on this (unfortunately true rates are hard to get due to many unknown cases)
# Symptom to hospitalisation is 5.76 days on average (Imperial #8)

infToHospitalExtra = np.array([1e-4, 1e-3, 2e-2, 1e-2])

# We do know at least how age affects these risks:

# For calculations see data_cleaning_py.ipynb, calculations from CHESS dataset as per 05 Apr
relativeAdmissionRisk_given_COVID_by_age = np.array([-0.94886625, -0.96332087, -0.86528671, -0.79828999, -0.61535305,
       -0.35214767,  0.12567034,  0.85809052,  3.55950368])

riskOfAEAttandance_by_age = np.array([0.41261361, 0.31560648, 0.3843979 , 0.30475704, 0.26659415,
       0.25203475, 0.24970244, 0.31549102, 0.65181376])


# Build the transition tensor from any non-hospitalised state to a hospitalised state
# (being in home quarantine is assumed to affect only the infection probability [below], not the hospitalisation probability)
# caseIsolationHospitalisationRateAdjustment = 1.

# This function takes as input the number of people in given age and health state, and in any non-hospitalised state
# and returns the number of people staying in the same age and health state,
# but now hospitalised (the rest of people remain in whatever state they were in)

def trFunc_HospitalAdmission(
    ageHospitalisationRateBaseline = ageHospitalisationRateBaseline,
    infToHospitalExtra = infToHospitalExtra,
    ageRelativeExtraAdmissionRiskToCovid = relativeAdmissionRisk_given_COVID_by_age * riskOfAEAttandance_by_age,

    **kwargs
    ):

    # This tensor will pointwise multiply an nAge x nHS slice of the stateTensor
    trTensor_HospitalAdmission = np.zeros((nAge, nHS))

    ageAdjusted_infToHospitalExtra = copy.deepcopy(np.repeat(infToHospitalExtra[np.newaxis],nAge,axis=0))
    for ii in range(ageAdjusted_infToHospitalExtra.shape[1]):
        # Adjust death rate by age dependent disease severity
        ageAdjusted_infToHospitalExtra[:,ii] = adjustRatesByAge_KeepAverageRate(
            infToHospitalExtra[ii],
            ageRelativeAdjustment=ageRelativeExtraAdmissionRiskToCovid
        )

    # Add baseline hospitalisation to all non-dead states
    trTensor_HospitalAdmission[:,:-1] += np.expand_dims(ageHospitalisationRateBaseline,-1)

    # Add COVID-caused hospitalisation to all infected states (TODO: This is summation of rates for independent processes, should be correct, but check)
    trTensor_HospitalAdmission[:,1:(nI+1)] += ageAdjusted_infToHospitalExtra

    return trTensor_HospitalAdmission


def trFunc_HospitalDischarge(
    ageHospitalisationRecoveryRateBaseline = ageHospitalisationRecoveryRateBaseline,
    dischargeDueToCovidRateMultiplier = 3.,

    **kwargs
    ):

    trTensor_HospitalDischarge = np.zeros((nAge, nHS))

    # Baseline discharges apply to all non-symptomatic patients (TODO: take into account testing state!)
    trTensor_HospitalDischarge[:, :3] += ageHospitalisationRecoveryRateBaseline[:,np.newaxis]

    # No discharges for COVID symptomatic people from the hospital until they recover
    # TODO - check with health experts if this is correct assumption; probably also depends on testing state
    trTensor_HospitalDischarge[:, 3:5] = 0.
    trTensor_HospitalDischarge[:, 5:7] = dischargeDueToCovidRateMultiplier * ageHospitalisationRecoveryRateBaseline[:,np.newaxis]

    return trTensor_HospitalDischarge

ageSocialMixingBaseline = pd.read_csv('../data/socialcontactdata_UK_Mossong2008_social_contact_matrix.csv', sep=',').iloc[:,1:].values


ageSocialMixingBaseline = (ageSocialMixingBaseline+ageSocialMixingBaseline.T)/2.

ageSocialMixingDistancing = pd.read_csv('../data/socialcontactdata_UK_Mossong2008_social_contact_matrix_with_distancing.csv', sep=',').iloc[:,1:].values

ageSocialMixingDistancing = (ageSocialMixingDistancing+ageSocialMixingDistancing.T)/2.

ageSocialMixingIsolation = np.zeros_like(ageSocialMixingBaseline)

elevatedMixingRatioInHospital = 3.0

withinHospitalSocialMixing = elevatedMixingRatioInHospital * np.sum(np.dot(agePopulationRatio, ageSocialMixingBaseline))

transmissionInfectionStage = np.array([0.001, 0.1, 0.6, 0.5])

def trFunc_newInfections_Complete(
    stateTensor,
    policySocialDistancing, # True / False, no default because it's important to know which one we use at any moment!
    policyImmunityPassports, # True / False, no default because it's important to know which one we use at any moment!
    ageSocialMixingBaseline = ageSocialMixingBaseline,
    ageSocialMixingDistancing = ageSocialMixingDistancing,
    ageSocialMixingIsolation = ageSocialMixingIsolation,
    withinHospitalSocialMixing = withinHospitalSocialMixing,
    transmissionInfectionStage = transmissionInfectionStage,

    **kwargs
):

    ageIsoContractionRate = np.zeros((nAge, nIso, nTest))


    # Add non-hospital infections
    #--------------------------------

    curNonIsolatedSocialMixing = ageSocialMixingDistancing if policySocialDistancing else ageSocialMixingBaseline

    # Add baseline interactions only between non-isolated people
    for k1 in [0,3]:
        for k2 in [0,3]:
            ageIsoContractionRate[:,k1,:] += np.expand_dims(
                np.matmul(
                    curNonIsolatedSocialMixing,
                    np.einsum('ijl,j->i',
                        stateTensor[:,1:(nI+1),k2,:], transmissionInfectionStage) # all infected in non-isolation
                ),
                axis=1
            )

    if policyImmunityPassports:
        # If the immunity passports policy is on, everyone who tested antibody positive, can roam freely
        # Therefore replace the interactions between people with testingState = 2 with ageSocialMixingBaseline
        # we do this by using the distributive property of matrix multiplication, and adding extra interactions
        # "ageSocialMixingBaseline"-"curNonIsolatedSocialMixing" with each other (this is zero if no social distancing!)
        # TODO - this is a bit hacky?, but probably correct - double check though!
        for k1 in [0,3]:
            for k2 in [0,3]:
                ageIsoContractionRate[:,k1,2:] += np.matmul(
                        ageSocialMixingBaseline-curNonIsolatedSocialMixing,
                        np.einsum('ijk,j->ik',
                            stateTensor[:,1:(nI+1),k2,2:], transmissionInfectionStage) # all infected in non-isolation
                    )
    # Add isolation interactions only between isolated and non-isolated people
    # non-isolated contracting it from isolated
    for k1 in [0,3]:
        ageIsoContractionRate[:,k1,:] += np.expand_dims(
            np.matmul(
                ageSocialMixingIsolation,
                np.einsum('ijl,j->i',
                    stateTensor[:,1:(nI+1),1,:], transmissionInfectionStage) # all infected in isolation
            ),
            axis=1
        )
    # isolated contracting it from non-isolated
    for k1 in [0,3]:
        ageIsoContractionRate[:,1,:] += np.expand_dims(
            np.matmul(
                ageSocialMixingIsolation,
                np.einsum('ijl,j->i',
                    stateTensor[:,1:(nI+1),k1,:], transmissionInfectionStage) # all infected in non-hospital, non-isolation
            ),
            axis = 1
        )

        # isolated cannot contracting it from another isolated


    # Add in-hospital infections (of hospitalised patients, and staff)
    #--------------------------------
    # (TODO - within hospitals we probably want to take into effect the testing state;
    #      tested people are better isolated and there's less mixing)

    ageIsoContractionRate[:,2:,:] += np.expand_dims(
            withinHospitalSocialMixing *
            np.einsum('ijkl,j->i',
                stateTensor[:,1:(nI+1),2:,:], transmissionInfectionStage), # all infected in hospital (sick or working)
        axis = (1,2))


    return ageIsoContractionRate/np.sum(stateTensor) # Normalise the rate by total population

def trFunc_travelInfectionRate_ageAdjusted(
    t, # Time (int, in days) within simulation

    travelMaxTime = 200,
    travelBaseRate = 5e-4, # How many people normally travel back to the country per day # TODO - get data
    travelDecline_mean = 15.,
    travelDecline_slope = 1.,

    travelInfection_peak = 1e-1,
    travelInfection_maxloc = 10.,
    travelInfection_shape = 2.,

    **kwargs
):

    tmpTime = np.arange(travelMaxTime)
    # nAge x T TODO get some realistic data on this
    travelAgeRateByTime = travelBaseRate * np.outer(agePopulationRatio, 1-expit((tmpTime-travelDecline_mean)/travelDecline_slope))


    # 1 x T TODO get some realistic data on this, maybe make it age weighted
    travelContractionRateByTime = stats.gamma.pdf(tmpTime, a=travelInfection_shape, loc=0., scale=travelInfection_maxloc/(travelInfection_shape-1))
    travelContractionRateByTime = travelInfection_peak*travelContractionRateByTime/np.max(travelContractionRateByTime)


    if t >= travelAgeRateByTime.shape[-1]:
        return np.zeros(travelAgeRateByTime.shape[0])
    else:
        return travelAgeRateByTime[:,int(t)] * travelContractionRateByTime[int(t)]

def inpFunc_testSpecifications(
    PCR_FNR_I1_to_R2 = np.array([ 0.9,  0.4, 0.15, 0.35, 0.5, 0.8]),
    PCR_FPR = 0.01,
    antigen_FNR_I1_to_R2 = np.array([ 0.95, 0.6, 0.35, 0.45, 0.6, 0.9]),
    antigen_FPR = 0.1,
    antibody_FNR_I1_to_R2 = np.array([0.99, 0.85, 0.8, 0.65, 0.3, 0.05]),
    antibody_FPR_S_to_I4 =  np.array([0.05, 0.04, 0.03, 0.02, 0.01])
    ):


    testSpecifications = pd.DataFrame(
    columns=["Name"],#, "Infection stage"],#, "Sensitivity", "Specificity"],

    data = (
        ["PCR"] * nHS +
        ["Antigen"] * (nHS) +
        ["Antibody"] * (nHS))
    )

    testSpecifications['OutputTestState'] = [1]*nHS + [1]*nHS + [2]*nHS # what information state does a pos test transition you to.

    testSpecifications['TruePosHealthState'] = [np.arange(1,nI+1)]*nHS + [np.arange(1,nI+1)]*nHS + [np.arange(nI+1,nI+nR+1)]*nHS # what information state does a pos test transition you to.

    # In some health states some people are true negatives and some are true positives! (No, makes litte sense to use, just account for it in FPR? Only matters for test makers...)
    # testSpecifications['AmbiguousPosHealthState'] = [np.arange(nI+1, nI+nR+1)]*nHS + [np.arange(nI+1, nI+nR+1)]*nHS + [np.arange(1, nI+1)]*nHS # what information state does a pos test transition you to.

    testSpecifications['InputHealthState'] = list(np.tile(range(nHS),3))

    # These numbers below are "defaults" illustrating the concept, but are modified by the inputs!!!

    testSpecifications['FalseNegativeRate'] = [ # ratio of positive (infected / immune) people missed by the test
        # For each health stage:
        #  S -> I1 (asymp) -> I2 (mild symp) -> I3 (symp, sick) -> I4 (symp, less sick) -> R1 / R2 (IgM, IgG avail) -> D

        # PCR
            0.,   0.9,            0.4,           0.15,                0.35,              0.5, 0.8,   0.,

        # Antigen
            0.,   0.95,           0.6,           0.35,                0.45,              0.6, 0.9,   0.,

        # Antibody
            0.,   0.99,           0.85,          0.8,                 0.65,              0.3, 0.05,  0.
    ]


    testSpecifications.loc[1:6,'FalseNegativeRate'] = PCR_FNR_I1_to_R2
    testSpecifications.loc[9:14,'FalseNegativeRate'] = antigen_FNR_I1_to_R2
    testSpecifications.loc[17:22,'FalseNegativeRate'] = antibody_FNR_I1_to_R2



    testSpecifications['FalsePositiveRate'] = [ # ratio of negative (non-infected or not immune) people deemed positive by the test
        # PCR
        0.01, 0.,0.,0.,0., 0.01, 0.01, 0.,

        # Antigen
        0.1, 0.,0.,0.,0., 0.1, 0.1, 0.,

        # Antibody
        0.05, 0.04, 0.03, 0.02, 0.01, 0., 0., 0.
    ]

    testSpecifications.loc[0,'FalsePositiveRate'] = PCR_FPR
    testSpecifications.loc[5:6,'FalsePositiveRate'] = PCR_FPR
    testSpecifications.loc[8,'FalsePositiveRate'] = antigen_FPR
    testSpecifications.loc[13:14,'FalsePositiveRate'] = antigen_FPR
    testSpecifications.loc[16:20,'FalsePositiveRate'] = antibody_FPR_S_to_I4

    name = testSpecifications['Name']
    truePosHealthState = testSpecifications['TruePosHealthState']
    testSpecifications.drop(['Name', 'TruePosHealthState'], inplace=True, axis=1)
    testSpecifications = testSpecifications.to_numpy()
    name = name.to_numpy()
    truePosHealthState = truePosHealthState.to_numpy()
    return testSpecifications, name, truePosHealthState

def trFunc_testCapacity(
    realTime, # time within simulation (day)

    # PCR capacity - initial
    testCapacity_pcr_phe_total = 1e4,
    testCapacity_pcr_phe_inflexday = pd.to_datetime("2020-03-25", format="%Y-%m-%d"),
    testCapacity_pcr_phe_inflexslope = 5.,

    # PCR capacity - increased
    testCapacity_pcr_country_total = 1e5,
    testCapacity_pcr_country_inflexday = pd.to_datetime("2020-04-25", format="%Y-%m-%d"),
    testCapacity_pcr_country_inflexslope = 10,

    # Antibody / antigen capacity
    testCapacity_antibody_country_firstday = pd.to_datetime("2020-04-25", format="%Y-%m-%d"),

    testCapacity_antibody_country_total = 5e6,
    testCapacity_antibody_country_inflexday = pd.to_datetime("2020-05-20", format="%Y-%m-%d"),
    testCapacity_antibody_country_inflexslope = 20,

    testCapacity_antigenratio_country = 0.7,

    **kwargs

):

    # Returns a dictionary with test names and number available at day "t"

    outPCR = (
        #phe phase
        testCapacity_pcr_phe_total * expit((realTime-testCapacity_pcr_phe_inflexday).days/testCapacity_pcr_phe_inflexslope)
        +
        #whole country phase
        testCapacity_pcr_country_total * expit((realTime-testCapacity_pcr_country_inflexday).days/testCapacity_pcr_country_inflexslope)
    )


    if realTime<testCapacity_antibody_country_firstday:
        outAntiTotal = 0.
    else:
        outAntiTotal = (
            testCapacity_antibody_country_total * expit((realTime-testCapacity_antibody_country_inflexday).days/testCapacity_antibody_country_inflexslope)
        )

    return {
        "PCR": outPCR,
        "Antigen": outAntiTotal*testCapacity_antigenratio_country,
        "Antibody": outAntiTotal*(1-testCapacity_antigenratio_country)
    }

# To test the function, in runtests.jl
py_rTime = pd.to_datetime("2020-05-25", format="%Y-%m-%d")
__trFunc_testCapacity = trFunc_testCapacity(py_rTime)


# PARAMETER DICTIONARIES AND TABLES
# -----------------------------------------------------------------------------------------

# Build the nested parameter/computation graph of a single function.
def build_paramDict(cur_func):
    """
    This function iterates through all inputs of a function,
    and saves the default argument names and values into a dictionary.

    If any of the default arguments are functions themselves, then recursively (depth-first) adds an extra field to
    the dictionary, named <funcName + "_params">, that contains its inputs and arguments.

    The output of this function can then be passed as a "kwargs" object to the highest level function,
    which will then pass the parameter values to the lower dictionary levels appropriately
    """

    paramDict = OrderedDict()

    allArgs = inspect.getfullargspec(cur_func)

    # Check if there are any default parameters, if no, just return empty dict
    if allArgs.defaults is None:
        return paramDict


    for argname, argval in zip(allArgs.args[-len(allArgs.defaults):], allArgs.defaults):
        # Save the default argument
        paramDict[argname] = argval
        # If the default argument is a function, inspect it for further

        if callable(argval):
            # print(argname)
            paramDict[argname+"_params"] = build_paramDict(argval)

    return paramDict

# Function that computes the right side of the non-lin model ODE
def dydt_Complete(t,
                  stateTensor_flattened, # Might be double the normal size (as first dimension) _withNewOnlyCopy, if debugReturnNewPerDay

                  realStartDate = pd.to_datetime("2020-02-20", format="%Y-%m-%d"),

                  # debug
                  debugTransition = False,
                  debugTimestep = False,
                  debugReturnNewPerDay = True, # Now implemented by default into state iteration

                  # Dimensions
                  nAge=nAge, nHS=nHS, nI=nI, nR=nR, nIso=nIso, nTest=nTest,

                  # Input functions and tensors
                  # ----------------------------

                  # Health state updates
                  trFunc_diseaseProgression = trFunc_diseaseProgression,
                  trFunc_newInfections = trFunc_newInfections_Complete,

                  # Initial incoming travel-based infections (before restrictions)
                  trFunc_travelInfectionRate_ageAdjusted = trFunc_travelInfectionRate_ageAdjusted,

                  # Hospitalisation and recovery
                  trFunc_HospitalAdmission = trFunc_HospitalAdmission,
                  trFunc_HospitalDischarge = trFunc_HospitalDischarge,

                  # Policy changes (on social distancing for now) (TODO - possibly make more changes)
                  tStartSocialDistancing = pd.to_datetime("2020-03-23", format="%Y-%m-%d"),
                  tStopSocialDistancing = pd.to_datetime("2025-03-23", format="%Y-%m-%d"),

                  tStartImmunityPassports = pd.to_datetime("2025-03-23", format="%Y-%m-%d"),
                  tStopImmunityPassports = pd.to_datetime("2025-03-23", format="%Y-%m-%d"),

                  tStartQuarantineCaseIsolation = pd.to_datetime("2025-03-23", format="%Y-%m-%d"),
                  tStopQuarantineCaseIsolation = pd.to_datetime("2025-03-23", format="%Y-%m-%d"),
                #  trFunc_quarantine = trFunc_quarantine_caseIsolation,

                  # Testing
                  #trFunc_testing = trFunc_testing,
                  #policyFunc_testing = policyFunc_testing_symptomaticOnly,
                  #testSpecifications = testSpecifications,
                  #trFunc_testCapacity = trFunc_testCapacity,
                  #trFunc_testCapacity_param_testCapacity_antigenratio_country = 0.3

                  **kwargs

):
    _einsum4_test = None
    _einsum5_test, _einsum5_test1 = None, None
    _einsum9_test, _einsum9_test1 = None, None

    if debugTimestep:
        print(t)

    # Initialise return
    if debugReturnNewPerDay: # the input has 2 copies of the state tensor, second copy being the cumulative incomings
        stateTensor = np.reshape(stateTensor_flattened, [2, nAge, nHS, nIso, nTest])[0]
    else:
        stateTensor = np.reshape(stateTensor_flattened, [nAge, nHS, nIso, nTest])

    dydt = np.zeros_like(stateTensor)

    # Initialise the full transition tensor
    trTensor_complete = np.zeros((nAge, nHS, nIso, nTest, nHS, nIso, nTest))

    # Disease condition updates
    # ---------------------------
    trTensor_diseaseProgression = trFunc_diseaseProgression(**kwargs["trFunc_diseaseProgression_params"])
    _trTensor_diseaseProgression = trTensor_diseaseProgression
    # Get disease condition updates with no isolation or test transition ("diagonal along those")
    for k1 in [0,1,2,3]:
        np.einsum('ijlml->ijlm',
            trTensor_complete[:,:,k1,:,:,k1,:])[:] += np.expand_dims(
                trTensor_diseaseProgression[:,:,k1,:]
                ,[2]) # all non-hospitalised disease progression is same

    _einsum4_test = trTensor_complete # trTensor_complete after einsum

#     # Compute new infections (0->1 in HS) with no isolation or test transition ("diagonal along those")
#     cur_policySocialDistancing = (
#                     t >= (tStartSocialDistancing - realStartDate).days
#                 )*(
#                     t <   (tStopSocialDistancing - realStartDate).days
#                 )
#     cur_policyImmunityPassports = (
#                     t >= (tStartImmunityPassports - realStartDate).days
#                 )*(
#                     t <   (tStopImmunityPassports - realStartDate).days
#                 )
#     np.einsum('iklkl->ikl',
#         trTensor_complete[:,0,:,:,1,:,:])[:] += (
#             trFunc_newInfections(
#                 stateTensor,
#                 policySocialDistancing = cur_policySocialDistancing,
#                 policyImmunityPassports = cur_policyImmunityPassports,
#                 **kwargs["trFunc_newInfections_params"]
#             ))

    # Also add new infected from travelling of healthy people, based on time-within-simulation (this is correct with all (0,0) states, as tested or isolated people dont travel)
#     trTensor_complete[:,0,0,0,1,0,0] += trFunc_travelInfectionRate_ageAdjusted(t, **kwargs["trFunc_travelInfectionRate_ageAdjusted_params"])


    # Hospitalisation state updates
    # -----------------------

    # Hospitalisation and recovery rates
    # We assume for now that these only depend on age and disease progression, not on testing state
    # (TODO - update this given new policies)

    # The disease and testing states don't change due to hospitalisation.
    # Hospital staff is treated as already hospitalised from all aspects expect social mixing, should suffice for now
    # TODO - Could try to devise a scheme in which hospital staff gets hospitalised and some recoveries from hospitalised state go back to hospital staff.
    # TODO - same issue with hospital staff home isolating; that's probably more important question!
#     for k1 in [0,1]:
#          np.einsum('ijljl->ijl',
#             trTensor_complete[:,:,k1,:,:,2,:])[:] += np.expand_dims(
#              trFunc_HospitalAdmission(**kwargs["trFunc_HospitalAdmission_params"]),[2])

#     # Add recovery from hospital rates
#     # TODO - again here (for now) we assume all discharged people go back to "normal state" instead of home isolation, have to think more on this
#     np.einsum('ijljl->ijl',
#             trTensor_complete[:,:,2,:,:,0,:])[:] += np.expand_dims(
#                  trFunc_HospitalDischarge(**kwargs["trFunc_HospitalDischarge_params"]),[2])





    # Testing state updates
    # ---------------------

    # trFunc_testing returns a stateTensor x testStates output
    #      after the policyFunc assigns tests that are evaluated according to testSpecifications

    # Diagonal (no transitions) in age, health state and isolation state
    # (for now, probably TODO: testing positive correlates with new hospitalisation!)
#     trTensor_testing = trFunc_testing(
#                                             stateTensor,
#                                             t,
#                                             realStartDate,
#                                             **kwargs["trFunc_testing_params"]
#                                         )

#     np.einsum('ijkljkm->ijklm',
#             trTensor_complete)[:] += trTensor_testing


    # Quarantine policy
    # ------------------

    # Check if policy is "on"
#     if (
#             t >= (tStartQuarantineCaseIsolation - realStartDate).days
#         )*(
#             t <   (tStopQuarantineCaseIsolation - realStartDate).days
#         ):
#         # New quarantining only happens to people who are transitioning already from untested to virus positive state
#         # Therefore here we DO use non-diagonal transitions, and we
#         #     redistribute the transtion rates given the testing (which was previously assumed not to create transition in isolation state)
#         trTensor_complete = trFunc_quarantine(
#                                                 trTensor_complete,
#                                                 t,
#                                                 trTensor_testing,
#                                                 **kwargs["trFunc_quarantine_params"]
#                                             )




    # Final corrections
    # -----------------



    # TODO: simulate aging and normal birth / death (not terribly important on these time scales, but should be quite simple)


    # Ensure that every "row" sums to 0 by adding to the diagonal (doesn't create new people out of nowhere)
    # Extract (writable) diagonal array and subtract the "row"-sums for each initial state
    np.einsum('ijkljkl->ijkl', trTensor_complete)[:] -= np.einsum('...jkl->...', trTensor_complete)

    _einsum9_test = trTensor_complete # State of trTensor_complete after 9th einsum

    # Compute the actual derivatives
    dydt = np.einsum('ijkl,ijklmnp->imnp', stateTensor, trTensor_complete) # contract the HS axis, keep age
    _einsum5_test = dydt # State of dydt after 5th einsum

    if debugReturnNewPerDay:
        """
            If this is true, instead of returning the real dydt,
            return only the positive "incoming" number of people to each state, so we can track "new cases"
            This needs some approximations, as follows:
                1. Take the normal transition tensor (with rates potentially > 0)
                2. From all states re-normalise the outgoing rates to sum at most to 1
                    (if they were less, keep it, if larger, then this represents
                    “in this day, all people will leave this state, in these ratios to these states”)
                3. Multiply only these outgoing rates with the current state
                    (so the result wont keep the same number of people as normal,
                    but only represent the “new incomings” for each state)
        """

        trTensor_complete_newOnly = copy.deepcopy(trTensor_complete)

        # TODO - Think - this is probably unnecessary actually, artifically reduces "new" rates?
#         # Devide each row by the absolute diagonal rate (that is the sum of the row), but only if its larger than 1
#         trTensor_complete_newOnly /= (
#             np.expand_dims(
#                 np.clip(np.abs(np.einsum('ijkljkl->ijkl', trTensor_complete_newOnly)), a_min=1., a_max=np.inf),
#                 axis=[4,5,6]
#             )
#         )

        # Set the diagonals to zero (no preservation, no outgoing, will end up being the incoming only)
        np.einsum('ijkljkl->ijkl', trTensor_complete_newOnly)[:] = 0.

        _einsum9_test1 = trTensor_complete_newOnly # State of trTensor_complete_newOnly after 9th einsum

        dydt_newOnly = np.einsum('ijkl,ijklmnp->imnp', stateTensor, trTensor_complete_newOnly)
        _einsum5_test1 = dydt_newOnly # State of trTensor_complete_newOnly after 5th einsum

        dydt = np.stack([dydt, dydt_newOnly], axis=0)


    if debugTransition:
        return np.reshape(dydt, -1), _trTensor_diseaseProgression, \
            _einsum4_test, _einsum5_test, _einsum5_test1, _einsum9_test, _einsum9_test1

    return np.reshape(dydt, -1), _trTensor_diseaseProgression, \
            _einsum4_test, _einsum5_test, _einsum5_test1, _einsum9_test, _einsum9_test1

# To test the function `dydt_Complete`
stateTensor_init=50.0*np.ones([nAge, nHS, nIso, nTest])
paramDict_default = build_paramDict(dydt_Complete)
paramDict_default["dydt_Complete"] = dydt_Complete
paramDict_default["INIT_stateTensor_init"] = stateTensor_init
# Example way to set parameters conveniently, here we start quarantining early based on test results
paramDict_current = copy.deepcopy(paramDict_default)
paramDict_current["tStartQuarantineCaseIsolation"] = pd.to_datetime("2020-03-23", format="%Y-%m-%d")

paramDict_current["debugReturnNewPerDay"]=False

state = 50*np.ones(9*8*4*4)
out, _trTensor_diseaseProgression, _einsum4_test, \
    _einsum5_test, _einsum5_test1, _einsum9_test, _einsum9_test1 \
        = dydt_Complete(0, state, **paramDict_current)

def f_symptoms_nonCOVID(
    realTime,
    symptomsIliRCGP = 15./100000., # Symptom rate in general non-hospitalised population
    symptomsRespInHospitalFAEs = 1.1/17.1, # Symptom rate in hospitalised population

    **kwargs):
    """
    This function defines the non-COVID ILI symptoms rate in the population at a given t time
    """


    # TODO, add extra data etc as input. For now:
    return (symptomsIliRCGP, symptomsRespInHospitalFAEs)



# In[18]:


# Distribute tests amongst (a given subset of) symptomatic people
def distTestsSymp(people, testsAvailable, noncovid_sympRatio, symp_HS = range(3,5), alreadyTestedRate = None):
    """
    distribute tests amongst symptomatic people
    people is nAge x nHS-1 x ... (excluding dead)
    """

    # Calculate noncovid, but symptomatic people
    peopleSymp = copy.deepcopy(people)
    peopleSymp[:, :min(symp_HS)] *= noncovid_sympRatio
    peopleSymp[:, max(symp_HS):] *= noncovid_sympRatio

    # Subtract already tested people
    if alreadyTestedRate is not None:
        peopleSymp -= people*alreadyTestedRate




    # Check if we already tested everyone with a different test
    if np.sum(peopleSymp)<1e-6:  # avoid numerical instabilities
        return (0.,0.)

    testedRatio = min(1., testsAvailable/np.sum(peopleSymp))


    return (
        # test rate
        testedRatio * (peopleSymp/(people+1e-6)), # avoid dividing by zero
        # tests used to achieve this
        testedRatio * np.sum(peopleSymp)
    )


# In[19]:


# Testing policies (how to distribute available tests)
# ----------------------------------------------------

# Estimate at any one time how many people are getting tested (with which tests) from which health states
def policyFunc_testing_symptomaticOnly(
    stateTensor,
    realTime,

    # Test types (names correspoding to testSpecifications)
    testTypes, # = ["PCR", "Antigen", "Antibody"],

    # Test Capacity (dict with names above and numbers available on day t)
    testsAvailable, # = trFunc_testCapacity(t)

    # OPTIONAL ARGUMENTS (may be different for different policy functions, should come with defaults!)
    antibody_testing_policy = "hospworker_then_random",
    # This has these values (for now), {"none", "hospworker_then_random", "virus_positive_only", "virus_positive_only_hospworker_first"}

    # Baseline symptoms
    f_symptoms_nonCOVID = f_symptoms_nonCOVID,

    distributeRemainingToRandom = True,
    return_testsAvailable_remaining = False,

    **kwargs
    ):
    """
    Returns a rate distribution of available test types over age, health and isolation states
    (although age assumed not to matter here)
    """
    
    # Output nAge x nHS x nIso x nTest x len(testTypes) tensor
    out_testRate = np.zeros(stateTensor.shape+(len(testTypes),))

    # Testing capacity is testsAvailable

    # Get sympom ratio. [0] - general, [1] - hospitalised
    cur_noncovid_sympRatio = f_symptoms_nonCOVID(realTime, **kwargs["f_symptoms_nonCOVID_params"])

    # PCR testing
    # -----------

    # Hospitalised people get priority over PCR tests
    testRate, testsUsed = distTestsSymp(
        people = stateTensor[:,:-1,2,0], # hospitalised non-positive people, exclude tested and dead people
        testsAvailable = testsAvailable["PCR"],
        noncovid_sympRatio = cur_noncovid_sympRatio[1]
    )

    out_testRate[:,:-1,2,0, testTypes.index("PCR")] += testRate
    testsAvailable["PCR"] -= testsUsed

    # Prioritise hospital workers next:
    # TODO: check if we should do this? In UK policy there was a 15% max for hospital worker testing until ~2 April...
    testRate, testsUsed = distTestsSymp(
        people = stateTensor[:,:-1,3,0],
        testsAvailable = testsAvailable["PCR"],
        noncovid_sympRatio= cur_noncovid_sympRatio[0]
    )

    out_testRate[:,:-1,3,0, testTypes.index("PCR")] += testRate
    testsAvailable["PCR"] -= testsUsed

    # Distribute PCRs left over the other populations
    testRate, testsUsed = distTestsSymp(
        people = stateTensor[:,:-1,:2,0],
        testsAvailable = testsAvailable["PCR"],
        noncovid_sympRatio= cur_noncovid_sympRatio[0]
    )

    out_testRate[:,:-1,:2,0, testTypes.index("PCR")] += testRate
    testsAvailable["PCR"] -= testsUsed

    if distributeRemainingToRandom:
        # Distribute PCRs left over the other populations
        testRate, testsUsed = distTestsSymp(
            people = stateTensor[:,:-1,:,0],
            testsAvailable = testsAvailable["PCR"],
            noncovid_sympRatio= 1.,
            alreadyTestedRate= out_testRate[:,:-1,:,0, testTypes.index("PCR")]
        )

        out_testRate[:,:-1,:,0, testTypes.index("PCR")] += testRate
        testsAvailable["PCR"] -= testsUsed


    # Antigen testing
    # ---------------

    # Hospitalised people get priority over PCR tests
    testRate, testsUsed = distTestsSymp(
        people = stateTensor[:,:-1,2,0], # hospitalised non-positive people, exclude tested and dead people
        testsAvailable = testsAvailable["Antigen"],
        noncovid_sympRatio= cur_noncovid_sympRatio[1],
        alreadyTestedRate=out_testRate[:,:-1,2, 0, testTypes.index("PCR")]
    )

    out_testRate[:,:-1,2,0, testTypes.index("Antigen")] += testRate
    testsAvailable["Antigen"] -= testsUsed

    # Prioritise hospital workers next:
    # TODO: check if we should do this? In UK policy there was a 15% max for hospital worker testing until ~2 April...
    testRate, testsUsed = distTestsSymp(
        people = stateTensor[:,:-1,3,0],
        testsAvailable = testsAvailable["Antigen"],
        noncovid_sympRatio= cur_noncovid_sympRatio[0],
        alreadyTestedRate=out_testRate[:,:-1,3, 0, testTypes.index("PCR")]
    )

    out_testRate[:,:-1,3,0, testTypes.index("Antigen")] += testRate
    testsAvailable["Antigen"] -= testsUsed

    # Distribute Antigen tests left over the other symptomatic people
    testRate, testsUsed = distTestsSymp(
        people = stateTensor[:,:-1,:2,0],
        testsAvailable = testsAvailable["Antigen"],
        noncovid_sympRatio= cur_noncovid_sympRatio[0],
        alreadyTestedRate=out_testRate[:,:-1,:2, 0, testTypes.index("PCR")]
    )

    out_testRate[:,:-1,:2,0, testTypes.index("Antigen")] += testRate
    testsAvailable["Antigen"] -= testsUsed

    if distributeRemainingToRandom:
        # Distribute antigen tests left over the other non-symptmatic populations
        testRate, testsUsed = distTestsSymp(
            people = stateTensor[:,:-1,:,0],
            testsAvailable = testsAvailable["Antigen"],
            noncovid_sympRatio= 1.,
            alreadyTestedRate= out_testRate[:,:-1,:,0, :].sum(-1)
        )

        out_testRate[:,:-1,:,0, testTypes.index("Antigen")] += testRate
        testsAvailable["Antigen"] -= testsUsed


    # Antibody testing
    # ----------------

    if antibody_testing_policy == "hospworker_then_random":

        # For now: give to hospital workers first, not taking into account previous tests or symptoms
        testRate, testsUsed = distTestsSymp(
            people = stateTensor[:,:-1,3,:2],
            testsAvailable = testsAvailable["Antibody"],
            noncovid_sympRatio= 1. # basically workers get antibody tested regardless of symptoms
        )

        out_testRate[:,:-1,3,:2, testTypes.index("Antibody")] += testRate
        testsAvailable["Antibody"] -= testsUsed

        # Afterwards let's just distribute randomly in the rest of the population
        testRate, testsUsed = distTestsSymp(
            people = stateTensor[:,:-1,:3,:2],
            testsAvailable = testsAvailable["Antibody"],
            noncovid_sympRatio= 1. # basically people get antibody tested regardless of symptoms
        )

        out_testRate[:,:-1,:3,:2, testTypes.index("Antibody")] += testRate
        testsAvailable["Antibody"] -= testsUsed

    if antibody_testing_policy == "virus_positive_only_hospworker_first":

        # For now: give to hospital workers first, not taking into account previous tests or symptoms
        testRate, testsUsed = distTestsSymp(
            people = stateTensor[:,:-1,3,1],
            testsAvailable = testsAvailable["Antibody"],
            noncovid_sympRatio= 1. # basically workers get antibody tested regardless of symptoms
        )

        out_testRate[:,:-1,3,1, testTypes.index("Antibody")] += testRate
        testsAvailable["Antibody"] -= testsUsed

        # Afterwards let's just distribute randomly in the rest of the population
        # TODO: Maybe prioratise people who tested positive for the virus before???
        testRate, testsUsed = distTestsSymp(
            people = stateTensor[:,:-1,:3,1],
            testsAvailable = testsAvailable["Antibody"],
            noncovid_sympRatio= 1. # basically people get antibody tested regardless of symptoms
        )

        out_testRate[:,:-1,:3,1, testTypes.index("Antibody")] += testRate
        testsAvailable["Antibody"] -= testsUsed


    if antibody_testing_policy == "virus_positive_only":

        testRate, testsUsed = distTestsSymp(
            people = stateTensor[:,:-1,:,1],
            testsAvailable = testsAvailable["Antibody"],
            noncovid_sympRatio= 1. # basically people get antibody tested regardless of symptoms
        )

        out_testRate[:,:-1,:,1, testTypes.index("Antibody")] += testRate
        testsAvailable["Antibody"] -= testsUsed

    if antibody_testing_policy == "none":
        out_testRate += 0.
        testsAvailable["Antibody"] -= 0.



    if return_testsAvailable_remaining:
        return out_testRate, testsAvailable

    return out_testRate

__policyFunc_testing_symptomaticOnly = policyFunc_testing_symptomaticOnly(
  stateTensor,
  py_rTime,
  ["PCR", "Antigen", "Antibody"],
  trFunc_testCapacity(py_rTime),
  **build_paramDict(policyFunc_testing_symptomaticOnly)
)

__policyFunc_testing_symptomaticOnly = np.transpose(
                                        __policyFunc_testing_symptomaticOnly)

basic_policyFunc_params_modified = build_paramDict(policyFunc_testing_symptomaticOnly)
basic_policyFunc_params_modified["distributeRemainingToRandom"] = False
basic_policyFunc_params_modified["return_testsAvailable_remaining"] = True

__policyFunc_testing_symptomaticOnly_, _dd = policyFunc_testing_symptomaticOnly(
  stateTensor,
  py_rTime,
  ["PCR", "Antigen", "Antibody"],
  trFunc_testCapacity(py_rTime),
  **basic_policyFunc_params_modified
)
__policyFunc_testing_symptomaticOnly_ = np.transpose(__policyFunc_testing_symptomaticOnly_)
# print(__policyFunc_testing_symptomaticOnly_)

# Define reTesting policy(s) (ie give tests to people in non-0 test states!)
def policyFunc_testing_massTesting_with_reTesting(
    stateTensor,
    realTime,

    # Test types (names correspoding to testSpecifications)
    testTypes, # = ["PCR", "Antigen", "Antibody"],

    # Test Capacity (dict with names above and numbers available on day t)
    testsAvailable, # = trFunc_testCapacity(t)

    # OPTIONAL ARGUMENTS (may be different for different policy functions, should come with defaults!)

    basic_policyFunc = policyFunc_testing_symptomaticOnly,
    # This basic policy will:
    # - do PCRs on symptomatic hospitalised people
    # - do PCRs on symptomatic hospital staff
    # - do PCRs on symptomatic non-hospitalised people
    # If PCRs run out at any stage, we use antigen tests with same priorisation

    # Afterwards given fractions of remaining antigen tests are distributed amongst people given these ratios and their earlier testing status:
    #retesting_antigen_viruspos_ratio = 0.1, # find virus false positives
    # UPDATE <- retesting viruspos is same ratio is normal testing, as long as they're not in quarantine already!
    retesting_antigen_immunepos_ratio = 0.05, # find immunity false positives
    # The rest of antigen tests are given out randomly

    # Antibody tests are used primarily on people who tested positive for the virus
    #  (set in basic_policyFunc!, use "virus_positive_only_hospworker_first"!)
    # Afterwards we can use the remaining on either random people (dangerous with many false positives!)
    # or for retesting people with already positive immune tests to make sure they're still immune,
    # controlled by this ratio:
    retesting_antibody_immunepos_ratio = 1.,

    #distributeRemainingToRandom = True, # TODO - otherwise stockpile for future, how?
    return_testsAvailable_remaining = False,

    **kwargs
    ):

    # Output nAge x nHS x nIso x nTest x len(testTypes) tensor
    out_testRate = np.zeros(stateTensor.shape+(len(testTypes),))


    # First distribute tests to symptomatic people as usual:

    # inpArgs change to not distributing tests randomly:
    basic_policyFunc_params_modified = copy.deepcopy(kwargs["basic_policyFunc_params"])
    basic_policyFunc_params_modified["distributeRemainingToRandom"] = False
    basic_policyFunc_params_modified["return_testsAvailable_remaining"] = True

    # Run the basic policy function with these modified parameters
    out_testRate, testsAvailable = basic_policyFunc(
        stateTensor,
        realTime = realTime,
        testTypes = testTypes,
        testsAvailable = testsAvailable,
        **basic_policyFunc_params_modified
    )
    beforeAntigen = out_testRate

    # We assume PCRs tend to run out done on symptomatic people in 0 Test state, so no retesting via PCR.


    # Antigen testing
    # ---------------

    # Retesting immune positive people
    testRate, testsUsed = distTestsSymp(
        people = stateTensor[:,:-1,:,2:], # immune positive people
        testsAvailable = testsAvailable["Antigen"] * retesting_antigen_immunepos_ratio,
        noncovid_sympRatio= 1., # set to 1. for ignoring symptom vs non-symptom
    )

    out_testRate[:,:-1,:,2:, testTypes.index("Antigen")] += testRate
    testsAvailable["Antigen"] -= testsUsed


    # Distribute antigen tests left over the other non-symptmatic populations
    # UPDATE <- here we use tests equally distributed among people with negative or positive previous virus tests,
    # as long as they are in non-quarantined state (isoState 0) # TODO - hospital worker testing???
    testRate, testsUsed = distTestsSymp(
        people = stateTensor[:,:-1,0,:2], # non-quarantined virus positive people
        testsAvailable = testsAvailable["Antigen"],
        noncovid_sympRatio= 1.,
        alreadyTestedRate= out_testRate[:,:-1,0,:2, testTypes.index("Antigen")] + out_testRate[:,:-1,0,:2, testTypes.index("PCR")]
    )

    out_testRate[:,:-1,0,:2, testTypes.index("Antigen")] += testRate
    testsAvailable["Antigen"] -= testsUsed


    # Antibody testing
    # -----------------
    # Retesting antibody positive people
    testRate, testsUsed = distTestsSymp(
        people = stateTensor[:,:-1,:,2:], # virus positive people
        testsAvailable = testsAvailable["Antibody"] * retesting_antibody_immunepos_ratio,
        noncovid_sympRatio= 1., # set to 1. for ignoring symptom vs non-symptom
    )


    # Afterwards let's just distribute randomly in the rest of the population
    testRate, testsUsed = distTestsSymp(
        people = stateTensor[:,:-1,:,:2],
        testsAvailable = testsAvailable["Antibody"],
        noncovid_sympRatio= 1., # basically people get antibody tested regardless of symptoms
        alreadyTestedRate= out_testRate[:,:-1,:,:2, testTypes.index("Antibody")]
    )

    out_testRate[:,:-1,:,:2, testTypes.index("Antibody")] += testRate
    testsAvailable["Antibody"] -= testsUsed


    if return_testsAvailable_remaining:
        return out_testRate, testsAvailable

    return out_testRate, beforeAntigen

__policyFunc_testing_massTesting_with_reTesting, beforeAntigen = policyFunc_testing_massTesting_with_reTesting(
    stateTensor,
    py_rTime,
    ["PCR", "Antigen", "Antibody"],
    trFunc_testCapacity(py_rTime),
    **build_paramDict(policyFunc_testing_massTesting_with_reTesting)
)

__policyFunc_testing_massTesting_with_reTesting = np.transpose(__policyFunc_testing_massTesting_with_reTesting)
beforeAntigen = np.transpose(beforeAntigen)
