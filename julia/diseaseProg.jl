# Based on England data (CHESS and NHS England)
# I want a way to keep this as the "average" disease progression,
# but modify it such that old people have less favorable outcomes (as observed)
# But correspondingly I want people at lower risk to have more favorable outcome on average

# For calculations see data_cleaning_py.ipynb, calculations from NHS England dataset as per 05 Apr
relativeDeathRisk_given_COVID_by_age = [-0.99742186, -0.99728639, -0.98158438,
                                        -0.9830432 , -0.82983414, -0.84039294,
                                         0.10768979,  0.38432409,  5.13754904]

#ageRelativeDiseaseSeverity = np.array([-0.8, -0.6, -0.3, -0.3, -0.1, 0.1, 0.35, 0.4, 0.5])
# FIXED (above) - this is a guess, find data and fix
#ageRelativeRecoverySpeed = np.array([0.2]*5+[-0.1, -0.2, -0.3, -0.5])
#TODO - this is a guess, find data and fix
ageRelativeRecoverySpeed = zeros(9)# For now we make it same for everyone, makes calculations easier
# For calculations see data_cleaning_py.ipynb, calculations from NHS England dataset as per 05 Apr
caseFatalityRatioHospital_given_COVID_by_age = [0.00856164, 0.03768844, 0.02321319,
            0.04282494, 0.07512237, 0.12550367, 0.167096  , 0.37953452, 0.45757006]

agePopulationTotal = 1000*[8044.056, 7642.473, 8558.707, 9295.024, 8604.251,
                                      9173.465, 7286.777, 5830.635, 3450.616]

function _agePopulationRatio(agePopulationTotal)
    agePopulationTotal *= 55.98/66.27
    return agePopulationTotal/sum(agePopulationTotal)
end

agePopulationRatio = _agePopulationRatio(agePopulationTotal)

function trFunc_diseaseProgression(ageRelativeRecoverySpeed::Array=
                                           ageRelativeRecoverySpeed,
                caseFatalityRatioHospital_given_COVID_by_age::Array=
                        caseFatalityRatioHospital_given_COVID_by_age,
                        relativeDeathRisk_given_COVID_by_age::Array=
                               relativeDeathRisk_given_COVID_by_age,
                                          agePopulationRatio::Array=
                                                  agePopulationRatio,
                                          agePopulationTotal::Array=
                                                  agePopulationTotal,
                                nonsymptomatic_ratio::Float64 = 0.86,
                                   # number of days between measurable events
                                   infect_to_symptoms::Float64 = 5.0,
                                   #symptom_to_death = 16.;
                                  symptom_to_recovery::Float64= 10.0, # 20.5; #unrealiticly long for old people
                          symptom_to_hospitalisation::Float64 = 5.76,
                        hospitalisation_to_recovery::Float64 = 14.51,
                                       IgG_formation::Float64 = 15.0,
                                   # Age related parameters
                                   # for now we'll assume that all hospitalised cases are known (overall 23% of hospitalised COVID patients die. 9% overall case fatality ratio)
                                   # Unknown rates to estimate
                                 nonsymp_to_recovery::Float64 = 15.0,
                                      inverse_IS1_IS2::Float64 = 4.0;
                                                            kwargs...)
    # Now we have all the information to build the age-aware multistage SIR model transition matrix
    # The full transition tensor is a sparse map from the Age x HealthState x isolation state to HealthState,
    # and thus is a 4th order tensor itself, representing a linear mapping
    # from "number of people aged A in health state B and isolation state C to health state D.
    agePopulationRatio = _agePopulationRatio(agePopulationTotal)
    nAge, nHS, nIso = kwargs[:nAge], kwargs[:nHS], kwargs[:nIso]
    trTensor_diseaseProgression = zeros((nHS, nIso, nHS, nAge))
    print(size(trTensor_diseaseProgression))
    # Use basic parameters to regularise inputs
    E_IS1 = 1.0/infect_to_symptoms
    # Numbers nonsymptomatic is assumed to be 86% -> E->IN / E-IS1 = 0.86/0.14
    E_IN = 0.86/0.14 * E_IS1
    # Nonsymptomatic recovery
    IN_R1 = 1.0/nonsymp_to_recovery
    IS1_IS2  = 1.0/inverse_IS1_IS2
    IS2_R1 = 1.0/(symptom_to_recovery - inverse_IS1_IS2)
    R1_R2 = 1.0/IgG_formation

    # Disease progression matrix # TODO - calibrate (together with transmissionInfectionStage)
    # rows: from-state, cols: to-state (non-symmetric!)
    # - this represent excess deaths only, doesn't contain baseline deaths!

    # Calculate all non-serious cases that do not end up in hospitals.
    # Note that we only have reliable death data from hospitals (NHS England),
    # so we do not model people dieing outside hospitals
    diseaseProgBaseline = [
    # to: E,    IN,    IS1,   IS2,    R1,      R2,     D
          0.0  E_IN  E_IS1    0       0        0       0    # from E
          0    0      0       0     IN_R1      0       0    # from IN
          0    0      0    IS1_IS2    0        0       0    # from IS1
          0    0      0       0     IS2_R1     0       0    # from IS2
          0    0      0       0       0       R1_R2    0    # from R1
          0    0      0       0       0        0       0    # from R2
          0    0      0       0       0        0       0    # from D
    ]
    diseaseProgBaseline = transpose(diseaseProgBaseline)
    # TODO can be improved
    # vcat(fill.(x, v)...) ???
    ageAdjusted_diseaseProgBaseline = deepcopy(cat(repeat([diseaseProgBaseline],
                                                              nAge)..., dims=3))
    println(size(ageAdjusted_diseaseProgBaseline))
    # Modify all death and R1 rates:
    for ii in range(1, stop = size(ageAdjusted_diseaseProgBaseline)[2])
        # Adjust death rate by age dependent disease severity  ??? check the key args
        ageAdjusted_diseaseProgBaseline[end, ii, :] = adjustRatesByAge_KeepAverageRate(
                                            ageAdjusted_diseaseProgBaseline[end, ii, 1],
                             agePopulationRatio=_agePopulationRatio(agePopulationTotal),
                              ageRelativeAdjustment=relativeDeathRisk_given_COVID_by_age
                              )
        # Adjust recovery rate by age dependent recovery speed
        ageAdjusted_diseaseProgBaseline[end - 2, ii, :] = adjustRatesByAge_KeepAverageRate(
                                            ageAdjusted_diseaseProgBaseline[end - 2, ii, 1],
                                                      agePopulationRatio=agePopulationRatio,
                                             ageRelativeAdjustment=ageRelativeRecoverySpeed
                                             )
    end
    ageAdjusted_diseaseProgBaseline_Hospital = deepcopy(ageAdjusted_diseaseProgBaseline)
    # Calculate hospitalisation based rates, for which we do have data. Hospitalisation can end up with deaths
    # Make sure that the ratio of recoveries in hospital honour the case fatality ratio appropriately
    # IS2 -> death
    ageAdjusted_diseaseProgBaseline_Hospital[end, 4, :] =
                     ageAdjusted_diseaseProgBaseline_Hospital[end - 2, 4, :] .* ( # IS2 -> recovery
                                  caseFatalityRatioHospital_given_COVID_by_age./(  # multiply by cfr / (1-cfr) to get correct rate towards death
                            1 .-  caseFatalityRatioHospital_given_COVID_by_age) )


    #TODO - time to death might be incorrect overall without an extra delay state, especially for young people
    # Non-hospitalised disease progression
    for i1 in [1, 2, 4]
        trTensor_diseaseProgression[2:end, i1, 2:end, :] = ageAdjusted_diseaseProgBaseline
    end
    # hospitalised disease progression
    trTensor_diseaseProgression[2:end, 3, 2:end, :] = ageAdjusted_diseaseProgBaseline_Hospital
    return trTensor_diseaseProgression
end


# Population (data from Imperial #13 ages.csv/UK)
#agePopulationTotal = 1000*[8044.056, 7642.473, 8558.707, 9295.024, 8604.251,
#                                      9173.465, 7286.777, 5830.635, 3450.616]
#agePopulationTotal = 1000.*pd.read_csv("https://raw.githubusercontent.com/ImperialCollegeLondon/covid19model/master/data/ages.csv").iloc[3].values[2:]
# Currently: let's work with england population only instead of full UK, as NHS England + CHESS data is much clearer than other regions
#agePopulationTotal *= 55.98/66.27 # (google england/uk population 2018, assuming age dist is similar)
#agePopulationRatio = agePopulationTotal/sum(agePopulationTotal)


function adjustRatesByAge_KeepAverageRate(rate; agePopulationRatio=nothing,
                                                ageRelativeAdjustment::Array=nothing,
                                                maxOutRate::Float64=10.0)
    if rate == 0
        return fill(0, size(ageRelativeAdjustment))
    end
    if rate >= maxOutRate
        @warn("somthing 1")
        return rate*(fill(1, size(ageRelativeAdjustment)))
    end
    out = fill(0, size(ageRelativeAdjustment))
    out[1] = maxOutRate + 1
    while sum(out .>= maxOutRate) > 0
        corrFactor = sum(agePopulationRatio ./ (1 .+ ageRelativeAdjustment))
        out =  rate * (1 .+ ageRelativeAdjustment) * corrFactor
        if sum(out .>= maxOutRate) > 0
            @warn("something 2")
            tmp_mean = sum(ageRelativeAdjustment)/length(ageRelativeAdjustment)
            ageRelativeAdjustment = tmp_mean .+ sqrt(0.9)*(
                                            ageRelativeAdjustment .- tmp_mean)
        end
    end
    return out
end

