# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# refs() — Multi-Format Bibliographic References
# =============================================================================

const _RefEntry = @NamedTuple{
    key::Symbol, authors::String, year::Int, title::String,
    journal::String, volume::String, issue::String, pages::String,
    doi::String, isbn::String, publisher::String, entry_type::Symbol
}

const _REFERENCES = Dict{Symbol, _RefEntry}(
    # --- Input-Output Analysis ---
    :leontief1936 => (key=:leontief1936, authors="Leontief, Wassily W.", year=1936,
        title="Quantitative Input and Output Relations in the Economic System of the United States",
        journal="Review of Economics and Statistics", volume="18", issue="3", pages="105--125",
        doi="10.2307/1927837", isbn="", publisher="", entry_type=:article),
    :ghosh1958 => (key=:ghosh1958, authors="Ghosh, Ambica", year=1958,
        title="Input-Output Approach in an Allocation System",
        journal="Economica", volume="25", issue="97", pages="58--64",
        doi="10.2307/2550694", isbn="", publisher="", entry_type=:article),
    :rasmussen1956 => (key=:rasmussen1956, authors="Rasmussen, Poul Norregaard", year=1956,
        title="Studies in Inter-Sectoral Relations", journal="",
        volume="", issue="", pages="", doi="",
        isbn="", publisher="North-Holland", entry_type=:book),
    :hirschman1958 => (key=:hirschman1958, authors="Hirschman, Albert O.", year=1958,
        title="The Strategy of Economic Development", journal="",
        volume="", issue="", pages="", doi="",
        isbn="", publisher="Yale University Press", entry_type=:book),
    :dietzenbacher_los1998 => (key=:dietzenbacher_los1998,
        authors="Dietzenbacher, Erik and Los, Bart", year=1998,
        title="Structural Decomposition Techniques: Sense and Sensitivity",
        journal="Economic Systems Research", volume="10", issue="4", pages="307--324",
        doi="10.1080/09535319800000023", isbn="", publisher="", entry_type=:article),
    :miller_blair_2009 => (key=:miller_blair_2009,
        authors="Miller, Ronald E. and Blair, Peter D.", year=2009,
        title="Input-Output Analysis: Foundations and Extensions",
        journal="", volume="", issue="", pages="",
        doi="10.1017/CBO9780511626982", isbn="978-0-521-51713-3",
        publisher="Cambridge University Press", entry_type=:book),
    :baqaee_farhi_2019 => (key=:baqaee_farhi_2019,
        authors="Baqaee, David Rezza and Farhi, Emmanuel", year=2019,
        title="The Macroeconomic Impact of Microeconomic Shocks: Beyond Hulten's Theorem",
        journal="Econometrica", volume="87", issue="4", pages="1155--1203",
        doi="10.3982/ECTA15202", isbn="", publisher="", entry_type=:article),
    # --- VAR & Structural VAR ---
    :sims1980 => (key=:sims1980, authors="Sims, Christopher A.", year=1980,
        title="Macroeconomics and Reality", journal="Econometrica",
        volume="48", issue="1", pages="1--48", doi="10.2307/1912017",
        isbn="", publisher="", entry_type=:article),
    :lutkepohl2005 => (key=:lutkepohl2005, authors="L\\\"utkepohl, Helmut", year=2005,
        title="New Introduction to Multiple Time Series Analysis", journal="",
        volume="", issue="", pages="", doi="",
        isbn="978-3-540-40172-8", publisher="Springer", entry_type=:book),
    :blanchard_quah1989 => (key=:blanchard_quah1989, authors="Blanchard, Olivier Jean and Quah, Danny", year=1989,
        title="The Dynamic Effects of Aggregate Demand and Supply Disturbances",
        journal="American Economic Review", volume="79", issue="4", pages="655--673",
        doi="", isbn="", publisher="", entry_type=:article),
    :uhlig2005 => (key=:uhlig2005, authors="Uhlig, Harald", year=2005,
        title="What Are the Effects of Monetary Policy on Output? Results from an Agnostic Identification Procedure",
        journal="Journal of Monetary Economics", volume="52", issue="2", pages="381--419",
        doi="10.1016/j.jmoneco.2004.05.007", isbn="", publisher="", entry_type=:article),
    :antolin_diaz_rubio_ramirez2018 => (key=:antolin_diaz_rubio_ramirez2018,
        authors="Antol{\\'\\i}n-D{\\'\\i}az, Juan and Rubio-Ram{\\'\\i}rez, Juan F.", year=2018,
        title="Narrative Sign Restrictions for SVARs",
        journal="American Economic Review", volume="108", issue="10", pages="2802--2829",
        doi="10.1257/aer.20161852", isbn="", publisher="", entry_type=:article),
    :arias_rubio_ramirez_waggoner2018 => (key=:arias_rubio_ramirez_waggoner2018,
        authors="Arias, Jonas E. and Rubio-Ram{\\'\\i}rez, Juan F. and Waggoner, Daniel F.", year=2018,
        title="Inference Based on Structural Vector Autoregressions Identified with Sign and Zero Restrictions: Theory and Applications",
        journal="Econometrica", volume="86", issue="2", pages="685--720",
        doi="10.3982/ECTA14468", isbn="", publisher="", entry_type=:article),
    :mountford_uhlig2009 => (key=:mountford_uhlig2009,
        authors="Mountford, Andrew and Uhlig, Harald", year=2009,
        title="What Are the Effects of Fiscal Policy Shocks?",
        journal="Journal of Applied Econometrics", volume="24", issue="6", pages="960--992",
        doi="10.1002/jae.1079", isbn="", publisher="", entry_type=:article),
    :kilian1998 => (key=:kilian1998, authors="Kilian, Lutz", year=1998,
        title="Small-Sample Confidence Intervals for Impulse Response Functions",
        journal="Review of Economics and Statistics", volume="80", issue="2", pages="218--230",
        doi="10.1162/003465398557465", isbn="", publisher="", entry_type=:article),
    :kilian_lutkepohl2017 => (key=:kilian_lutkepohl2017,
        authors="Kilian, Lutz and L\\\"utkepohl, Helmut", year=2017,
        title="Structural Vector Autoregressive Analysis", journal="",
        volume="", issue="", pages="", doi="",
        isbn="978-1-107-19657-5", publisher="Cambridge University Press", entry_type=:book),
    # --- Bayesian VAR ---
    :litterman1986 => (key=:litterman1986, authors="Litterman, Robert B.", year=1986,
        title="Forecasting with Bayesian Vector Autoregressions---Five Years of Experience",
        journal="Journal of Business \\& Economic Statistics", volume="4", issue="1", pages="25--38",
        doi="10.1080/07350015.1986.10509491", isbn="", publisher="", entry_type=:article),
    :kadiyala_karlsson1997 => (key=:kadiyala_karlsson1997,
        authors="Kadiyala, K. Rao and Karlsson, Sune", year=1997,
        title="Numerical Methods for Estimation and Inference in Bayesian VAR-Models",
        journal="Journal of Applied Econometrics", volume="12", issue="2", pages="99--132",
        doi="10.1002/(SICI)1099-1255(199703)12:2<99::AID-JAE429>3.0.CO;2-A",
        isbn="", publisher="", entry_type=:article),
    # --- Local Projections ---
    :jorda2005 => (key=:jorda2005, authors="Jord\\`a, \\`Oscar", year=2005,
        title="Estimation and Inference of Impulse Responses by Local Projections",
        journal="American Economic Review", volume="95", issue="1", pages="161--182",
        doi="10.1257/0002828053828518", isbn="", publisher="", entry_type=:article),
    :stock_watson2018 => (key=:stock_watson2018,
        authors="Stock, James H. and Watson, Mark W.", year=2018,
        title="Identification and Estimation of Dynamic Causal Effects in Macroeconomics Using External Instruments",
        journal="Economic Journal", volume="128", issue="610", pages="917--948",
        doi="10.1111/ecoj.12593", isbn="", publisher="", entry_type=:article),
    :barnichon_brownlees2019 => (key=:barnichon_brownlees2019,
        authors="Barnichon, Regis and Brownlees, Christian", year=2019,
        title="Impulse Response Estimation by Smooth Local Projections",
        journal="Review of Economics and Statistics", volume="101", issue="3", pages="522--530",
        doi="10.1162/rest_a_00778", isbn="", publisher="", entry_type=:article),
    :auerbach_gorodnichenko2012 => (key=:auerbach_gorodnichenko2012,
        authors="Auerbach, Alan J. and Gorodnichenko, Yuriy", year=2012,
        title="Measuring the Output Responses to Fiscal Policy",
        journal="American Economic Journal: Economic Policy", volume="4", issue="2", pages="1--27",
        doi="10.1257/pol.4.2.1", isbn="", publisher="", entry_type=:article),
    :angrist_jorda_kuersteiner2018 => (key=:angrist_jorda_kuersteiner2018,
        authors="Angrist, Joshua D. and Jord\\`a, \\`Oscar and Kuersteiner, Guido M.", year=2018,
        title="Semiparametric Estimates of Monetary Policy Effects: String Theory Revisited",
        journal="Journal of Business \\& Economic Statistics", volume="36", issue="3", pages="371--387",
        doi="10.1080/07350015.2016.1204919", isbn="", publisher="", entry_type=:article),
    :plagborg_moller_wolf2021 => (key=:plagborg_moller_wolf2021,
        authors="Plagborg-M{\\o}ller, Mikkel and Wolf, Christian K.", year=2021,
        title="Local Projections and VARs Estimate the Same Impulse Responses",
        journal="Econometrica", volume="89", issue="2", pages="955--980",
        doi="10.3982/ECTA17813", isbn="", publisher="", entry_type=:article),
    :gorodnichenko_lee2020 => (key=:gorodnichenko_lee2020,
        authors="Gorodnichenko, Yuriy and Lee, Byoungchan", year=2020,
        title="Forecast Error Variance Decompositions with Local Projections",
        journal="Journal of Business \\& Economic Statistics", volume="38", issue="4", pages="921--933",
        doi="10.1080/07350015.2019.1610661", isbn="", publisher="", entry_type=:article),
    # --- Factor Models ---
    :bai_ng2002 => (key=:bai_ng2002, authors="Bai, Jushan and Ng, Serena", year=2002,
        title="Determining the Number of Factors in Approximate Factor Models",
        journal="Econometrica", volume="70", issue="1", pages="191--221",
        doi="10.1111/1468-0262.00273", isbn="", publisher="", entry_type=:article),
    :stock_watson2002 => (key=:stock_watson2002,
        authors="Stock, James H. and Watson, Mark W.", year=2002,
        title="Forecasting Using Principal Components from a Large Number of Predictors",
        journal="Journal of the American Statistical Association", volume="97", issue="460", pages="1167--1179",
        doi="10.1198/016214502388618960", isbn="", publisher="", entry_type=:article),
    # --- Unit Root Tests ---
    :dickey_fuller1979 => (key=:dickey_fuller1979,
        authors="Dickey, David A. and Fuller, Wayne A.", year=1979,
        title="Distribution of the Estimators for Autoregressive Time Series with a Unit Root",
        journal="Journal of the American Statistical Association", volume="74", issue="366a", pages="427--431",
        doi="10.1080/01621459.1979.10482531", isbn="", publisher="", entry_type=:article),
    :kpss1992 => (key=:kpss1992,
        authors="Kwiatkowski, Denis and Phillips, Peter C. B. and Schmidt, Peter and Shin, Yongcheol", year=1992,
        title="Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root",
        journal="Journal of Econometrics", volume="54", issue="1--3", pages="159--178",
        doi="10.1016/0304-4076(92)90104-Y", isbn="", publisher="", entry_type=:article),
    :phillips_perron1988 => (key=:phillips_perron1988,
        authors="Phillips, Peter C. B. and Perron, Pierre", year=1988,
        title="Testing for a Unit Root in Time Series Regression",
        journal="Biometrika", volume="75", issue="2", pages="335--346",
        doi="10.1093/biomet/75.2.335", isbn="", publisher="", entry_type=:article),
    :zivot_andrews1992 => (key=:zivot_andrews1992,
        authors="Zivot, Eric and Andrews, Donald W. K.", year=1992,
        title="Further Evidence on the Great Crash, the Oil-Price Shock, and the Unit-Root Hypothesis",
        journal="Journal of Business \\& Economic Statistics", volume="10", issue="3", pages="251--270",
        doi="10.1080/07350015.1992.10509904", isbn="", publisher="", entry_type=:article),
    :ng_perron2001 => (key=:ng_perron2001,
        authors="Ng, Serena and Perron, Pierre", year=2001,
        title="Lag Length Selection and the Construction of Unit Root Tests with Good Size and Power",
        journal="Econometrica", volume="69", issue="6", pages="1519--1554",
        doi="10.1111/1468-0262.00256", isbn="", publisher="", entry_type=:article),
    :johansen1991 => (key=:johansen1991, authors="Johansen, S{\\o}ren", year=1991,
        title="Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models",
        journal="Econometrica", volume="59", issue="6", pages="1551--1580",
        doi="10.2307/2938278", isbn="", publisher="", entry_type=:article),
    :engle_granger1987 => (key=:engle_granger1987,
        authors="Engle, Robert F. and Granger, Clive W. J.", year=1987,
        title="Co-Integration and Error Correction: Representation, Estimation, and Testing",
        journal="Econometrica", volume="55", issue="2", pages="251--276",
        doi="10.2307/1913236", isbn="", publisher="", entry_type=:article),
    # --- ARIMA ---
    :box_jenkins1970 => (key=:box_jenkins1970,
        authors="Box, George E. P. and Jenkins, Gwilym M.", year=1970,
        title="Time Series Analysis: Forecasting and Control", journal="",
        volume="", issue="", pages="", doi="",
        isbn="978-0-8162-1094-7", publisher="Holden-Day", entry_type=:book),
    :hyndman_khandakar2008 => (key=:hyndman_khandakar2008,
        authors="Hyndman, Rob J. and Khandakar, Yeasmin", year=2008,
        title="Automatic Time Series Forecasting: The forecast Package for R",
        journal="Journal of Statistical Software", volume="27", issue="3", pages="1--22",
        doi="10.18637/jss.v027.i03", isbn="", publisher="", entry_type=:article),
    # --- ARFIMA / long memory (EV-13) ---
    :sowell1992 => (key=:sowell1992, authors="Sowell, Fallaw", year=1992,
        title="Maximum Likelihood Estimation of Stationary Univariate Fractionally Integrated Time Series Models",
        journal="Journal of Econometrics", volume="53", issue="1--3", pages="165--188",
        doi="10.1016/0304-4076(92)90084-5", isbn="", publisher="", entry_type=:article),
    :geweke_porter_hudak1983 => (key=:geweke_porter_hudak1983,
        authors="Geweke, John and Porter-Hudak, Susan", year=1983,
        title="The Estimation and Application of Long Memory Time Series Models",
        journal="Journal of Time Series Analysis", volume="4", issue="4", pages="221--238",
        doi="10.1111/j.1467-9892.1983.tb00371.x", isbn="", publisher="", entry_type=:article),
    :robinson1995 => (key=:robinson1995, authors="Robinson, Peter M.", year=1995,
        title="Gaussian Semiparametric Estimation of Long Range Dependence",
        journal="Annals of Statistics", volume="23", issue="5", pages="1630--1661",
        doi="10.1214/aos/1176324317", isbn="", publisher="", entry_type=:article),
    :hosking1981 => (key=:hosking1981, authors="Hosking, J. R. M.", year=1981,
        title="Fractional Differencing",
        journal="Biometrika", volume="68", issue="1", pages="165--176",
        doi="10.1093/biomet/68.1.165", isbn="", publisher="", entry_type=:article),
    :jensen_nielsen2014 => (key=:jensen_nielsen2014,
        authors="Jensen, Andreas Noack and Nielsen, Morten \\O{}rregaard", year=2014,
        title="A Fast Fractional Difference Algorithm",
        journal="Journal of Time Series Analysis", volume="35", issue="5", pages="428--436",
        doi="10.1111/jtsa.12074", isbn="", publisher="", entry_type=:article),
    :durbin_koopman2012 => (key=:durbin_koopman2012,
        authors="Durbin, James and Koopman, Siem Jan", year=2012,
        title="Time Series Analysis by State Space Methods", journal="",
        volume="", issue="", pages="", doi="10.1093/acprof:oso/9780199641178.001.0001",
        isbn="978-0-19-964117-8", publisher="Oxford University Press", entry_type=:book),
    # --- GMM ---
    :hansen1982 => (key=:hansen1982, authors="Hansen, Lars Peter", year=1982,
        title="Large Sample Properties of Generalized Method of Moments Estimators",
        journal="Econometrica", volume="50", issue="4", pages="1029--1054",
        doi="10.2307/1912775", isbn="", publisher="", entry_type=:article),
    # --- SMM ---
    :ruge_murcia2012 => (key=:ruge_murcia2012, authors="Ruge-Murcia, Francisco J.", year=2012,
        title="Estimating Nonlinear DSGE Models by the Simulated Method of Moments",
        journal="Journal of Economic Dynamics and Control", volume="36", issue="6", pages="914--938",
        doi="10.1016/j.jedc.2012.01.008", isbn="", publisher="", entry_type=:article),
    :lee_ingram1991 => (key=:lee_ingram1991, authors="Lee, Bong-Soo and Ingram, Beth Fisher", year=1991,
        title="Simulation Estimation of Time-Series Models",
        journal="Journal of Econometrics", volume="47", issue="2--3", pages="197--205",
        doi="10.1016/0304-4076(91)90098-X", isbn="", publisher="", entry_type=:article),
    :duffie_singleton1993 => (key=:duffie_singleton1993, authors="Duffie, Darrell and Singleton, Kenneth J.", year=1993,
        title="Simulated Moments Estimation of Markov Models of Asset Prices",
        journal="Econometrica", volume="61", issue="4", pages="929--952",
        doi="10.2307/2951768", isbn="", publisher="", entry_type=:article),
    # --- Analytical Moments ---
    :hamilton1994 => (key=:hamilton1994, authors="Hamilton, James D.", year=1994,
        title="Time Series Analysis",
        journal="", volume="", issue="", pages="",
        doi="", isbn="0-691-04289-6", publisher="Princeton University Press", entry_type=:book),
    # --- Statistical Identification — Survey ---
    :lewis2025 => (key=:lewis2025,
        authors="Lewis, Daniel J.", year=2025,
        title="Identification Based on Higher Moments in Macroeconometrics",
        journal="Annual Review of Economics", volume="17", issue="", pages="665--693",
        doi="10.1146/annurev-economics-070124-051419", isbn="", publisher="", entry_type=:article),
    :lewis2021 => (key=:lewis2021,
        authors="Lewis, Daniel J.", year=2021,
        title="Identifying Shocks via Time-Varying Volatility",
        journal="Review of Economic Studies", volume="88", issue="6", pages="3086--3124",
        doi="10.1093/restud/rdab009", isbn="", publisher="", entry_type=:article),
    :lewis2022 => (key=:lewis2022,
        authors="Lewis, Daniel J.", year=2022,
        title="Robust Inference in Models Identified via Heteroskedasticity",
        journal="Review of Economics and Statistics", volume="104", issue="3", pages="510--524",
        doi="10.1162/rest_a_00977", isbn="", publisher="", entry_type=:article),
    :sentana_fiorentini2001 => (key=:sentana_fiorentini2001,
        authors="Sentana, Enrique and Fiorentini, Gabriele", year=2001,
        title="Identification, Estimation and Testing of Conditionally Heteroskedastic Factor Models",
        journal="Journal of Econometrics", volume="102", issue="2", pages="143--164",
        doi="10.1016/S0304-4076(01)00051-3", isbn="", publisher="", entry_type=:article),
    :gourieroux_monfort_renne2017 => (key=:gourieroux_monfort_renne2017,
        authors="Gourieroux, Christian and Monfort, Alain and Renne, Jean-Paul", year=2017,
        title="Statistical Inference for Independent Component Analysis: Application to Structural VAR Models",
        journal="Journal of Econometrics", volume="196", issue="1", pages="111--126",
        doi="10.1016/j.jeconom.2016.09.007", isbn="", publisher="", entry_type=:article),
    :keweloh2021 => (key=:keweloh2021,
        authors="Keweloh, Sascha A.", year=2021,
        title="A Generalized Method of Moments Estimator for Structural Vector Autoregressions Based on Higher Moments",
        journal="Journal of Business \\& Economic Statistics", volume="39", issue="3", pages="772--882",
        doi="10.1080/07350015.2020.1730858", isbn="", publisher="", entry_type=:article),
    :lanne_luoto2021 => (key=:lanne_luoto2021,
        authors="Lanne, Markku and Luoto, Jani", year=2021,
        title="GMM Estimation of Non-Gaussian Structural Vector Autoregression",
        journal="Journal of Business \\& Economic Statistics", volume="39", issue="1", pages="69--81",
        doi="10.1080/07350015.2019.1629940", isbn="", publisher="", entry_type=:article),
    :comon1994 => (key=:comon1994,
        authors="Comon, Pierre", year=1994,
        title="Independent Component Analysis, A New Concept?",
        journal="Signal Processing", volume="36", issue="3", pages="287--314",
        doi="10.1016/0165-1684(94)90029-9", isbn="", publisher="", entry_type=:article),
    :lanne_lutkepohl2008 => (key=:lanne_lutkepohl2008,
        authors="Lanne, Markku and L\\\"utkepohl, Helmut", year=2008,
        title="Identifying Monetary Policy Shocks via Changes in Volatility",
        journal="Journal of Money, Credit and Banking", volume="40", issue="6", pages="1131--1149",
        doi="10.1111/j.1538-4616.2008.00151.x", isbn="", publisher="", entry_type=:article),
    :normandin_phaneuf2004 => (key=:normandin_phaneuf2004,
        authors="Normandin, Michel and Phaneuf, Louis", year=2004,
        title="Monetary Policy Shocks: Testing Identification Conditions under Time-Varying Conditional Volatility",
        journal="Journal of Monetary Economics", volume="51", issue="6", pages="1217--1243",
        doi="10.1016/j.jmoneco.2003.11.002", isbn="", publisher="", entry_type=:article),
    # --- Non-Gaussian SVAR — ICA ---
    :hyvarinen1999 => (key=:hyvarinen1999, authors="Hyv\\\"arinen, Aapo", year=1999,
        title="Fast and Robust Fixed-Point Algorithms for Independent Component Analysis",
        journal="IEEE Transactions on Neural Networks", volume="10", issue="3", pages="626--634",
        doi="10.1109/72.761722", isbn="", publisher="", entry_type=:article),
    :cardoso_souloumiac1993 => (key=:cardoso_souloumiac1993,
        authors="Cardoso, Jean-Fran{\\c{c}}ois and Souloumiac, Antoine", year=1993,
        title="Blind Beamforming for Non-Gaussian Signals",
        journal="IEE Proceedings F --- Radar and Signal Processing", volume="140", issue="6", pages="362--370",
        doi="10.1049/ip-f-2.1993.0054", isbn="", publisher="", entry_type=:article),
    :belouchrani1997 => (key=:belouchrani1997,
        authors="Belouchrani, Adel and Abed-Meraim, Karim and Cardoso, Jean-Fran{\\c{c}}ois and Moulines, Eric",
        year=1997, title="A Blind Source Separation Technique Using Second-Order Statistics",
        journal="IEEE Transactions on Signal Processing", volume="45", issue="2", pages="434--444",
        doi="10.1109/78.554307", isbn="", publisher="", entry_type=:article),
    :szekely_rizzo_bakirov2007 => (key=:szekely_rizzo_bakirov2007,
        authors="Sz{\\'e}kely, G{\\'a}bor J. and Rizzo, Maria L. and Bakirov, Nail K.", year=2007,
        title="Measuring and Testing Dependence by Correlation of Distances",
        journal="Annals of Statistics", volume="35", issue="6", pages="2769--2794",
        doi="10.1214/009053607000000505", isbn="", publisher="", entry_type=:article),
    :matteson_tsay2017 => (key=:matteson_tsay2017,
        authors="Matteson, David S. and Tsay, Ruey S.", year=2017,
        title="Independent Component Analysis via Distance Covariance",
        journal="Journal of the American Statistical Association", volume="112", issue="518", pages="623--637",
        doi="10.1080/01621459.2016.1150851", isbn="", publisher="", entry_type=:article),
    :gretton2005 => (key=:gretton2005,
        authors="Gretton, Arthur and Bousquet, Olivier and Smola, Alex and Sch\\\"olkopf, Bernhard", year=2005,
        title="Measuring Statistical Dependence with Hilbert-Schmidt Norms",
        journal="Algorithmic Learning Theory", volume="3734", issue="", pages="63--77",
        doi="10.1007/11564089_7", isbn="", publisher="", entry_type=:incollection),
    # --- Non-Gaussian SVAR — ML ---
    :lanne_meitz_saikkonen2017 => (key=:lanne_meitz_saikkonen2017,
        authors="Lanne, Markku and Meitz, Mika and Saikkonen, Pentti", year=2017,
        title="Identification and Estimation of Non-Gaussian Structural Vector Autoregressions",
        journal="Journal of Econometrics", volume="196", issue="2", pages="288--304",
        doi="10.1016/j.jeconom.2016.06.002", isbn="", publisher="", entry_type=:article),
    # --- Non-Gaussian SVAR — Heteroskedasticity ---
    :rigobon2003 => (key=:rigobon2003, authors="Rigobon, Roberto", year=2003,
        title="Identification Through Heteroskedasticity",
        journal="Review of Economics and Statistics", volume="85", issue="4", pages="777--792",
        doi="10.1162/003465303772815727", isbn="", publisher="", entry_type=:article),
    :lutkepohl_netsunajev2017 => (key=:lutkepohl_netsunajev2017,
        authors="L\\\"utkepohl, Helmut and Netsunajev, Aleksei", year=2017,
        title="Structural Vector Autoregressions with Smooth Transition in Variances",
        journal="Journal of Economic Dynamics and Control", volume="84", issue="", pages="43--57",
        doi="10.1016/j.jedc.2017.09.001", isbn="", publisher="", entry_type=:article),
    # --- Normality Tests ---
    :jarque_bera1980 => (key=:jarque_bera1980,
        authors="Jarque, Carlos M. and Bera, Anil K.", year=1980,
        title="Efficient Tests for Normality, Homoscedasticity and Serial Independence of Regression Residuals",
        journal="Economics Letters", volume="6", issue="3", pages="255--259",
        doi="10.1016/0165-1765(80)90024-5", isbn="", publisher="", entry_type=:article),
    :mardia1970 => (key=:mardia1970, authors="Mardia, Kanti V.", year=1970,
        title="Measures of Multivariate Skewness and Kurtosis with Applications",
        journal="Biometrika", volume="57", issue="3", pages="519--530",
        doi="10.1093/biomet/57.3.519", isbn="", publisher="", entry_type=:article),
    :doornik_hansen2008 => (key=:doornik_hansen2008,
        authors="Doornik, Jurgen A. and Hansen, Henrik", year=2008,
        title="An Omnibus Test for Univariate and Multivariate Normality",
        journal="Oxford Bulletin of Economics and Statistics", volume="70", issue="s1", pages="927--939",
        doi="10.1111/j.1468-0084.2008.00537.x", isbn="", publisher="", entry_type=:article),
    :henze_zirkler1990 => (key=:henze_zirkler1990,
        authors="Henze, Norbert and Zirkler, Bernhard", year=1990,
        title="A Class of Invariant Consistent Tests for Multivariate Normality",
        journal="Communications in Statistics --- Theory and Methods", volume="19", issue="10", pages="3595--3617",
        doi="10.1080/03610929008830400", isbn="", publisher="", entry_type=:article),
    # --- Covariance Estimators ---
    :newey_west1987 => (key=:newey_west1987,
        authors="Newey, Whitney K. and West, Kenneth D.", year=1987,
        title="A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix",
        journal="Econometrica", volume="55", issue="3", pages="703--708",
        doi="10.2307/1913610", isbn="", publisher="", entry_type=:article),
    :white1980 => (key=:white1980, authors="White, Halbert", year=1980,
        title="A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity",
        journal="Econometrica", volume="48", issue="4", pages="817--838",
        doi="10.2307/1912934", isbn="", publisher="", entry_type=:article),
    # --- OLS Residual Diagnostics (EV-31, #439) ---
    :breusch_pagan1979 => (key=:breusch_pagan1979,
        authors="Breusch, Trevor S. and Pagan, Adrian R.", year=1979,
        title="A Simple Test for Heteroscedasticity and Random Coefficient Variation",
        journal="Econometrica", volume="47", issue="5", pages="1287--1294",
        doi="10.2307/1911963", isbn="", publisher="", entry_type=:article),
    :koenker1981 => (key=:koenker1981, authors="Koenker, Roger", year=1981,
        title="A Note on Studentizing a Test for Heteroscedasticity",
        journal="Journal of Econometrics", volume="17", issue="1", pages="107--112",
        doi="10.1016/0304-4076(81)90062-2", isbn="", publisher="", entry_type=:article),
    :glejser1969 => (key=:glejser1969, authors="Glejser, Herbert", year=1969,
        title="A New Test for Heteroskedasticity",
        journal="Journal of the American Statistical Association", volume="64", issue="325",
        pages="316--323", doi="10.1080/01621459.1969.10500976", isbn="", publisher="",
        entry_type=:article),
    :harvey1976 => (key=:harvey1976, authors="Harvey, Andrew C.", year=1976,
        title="Estimating Regression Models with Multiplicative Heteroscedasticity",
        journal="Econometrica", volume="44", issue="3", pages="461--465",
        doi="10.2307/1913974", isbn="", publisher="", entry_type=:article),
    :godfrey1978 => (key=:godfrey1978, authors="Godfrey, Leslie G.", year=1978,
        title="Testing Against General Autoregressive and Moving Average Error Models when the Regressors Include Lagged Dependent Variables",
        journal="Econometrica", volume="46", issue="6", pages="1293--1301",
        doi="10.2307/1913829", isbn="", publisher="", entry_type=:article),
    :breusch1978 => (key=:breusch1978, authors="Breusch, Trevor S.", year=1978,
        title="Testing for Autocorrelation in Dynamic Linear Models",
        journal="Australian Economic Papers", volume="17", issue="31", pages="334--355",
        doi="10.1111/j.1467-8454.1978.tb00635.x", isbn="", publisher="", entry_type=:article),
    :ramsey1969 => (key=:ramsey1969, authors="Ramsey, James B.", year=1969,
        title="Tests for Specification Errors in Classical Linear Least-Squares Regression Analysis",
        journal="Journal of the Royal Statistical Society, Series B", volume="31", issue="2",
        pages="350--371", doi="10.1111/j.2517-6161.1969.tb00796.x", isbn="", publisher="",
        entry_type=:article),
    # --- Stability & Influence Diagnostics (EV-32, #440) ---
    :brown_durbin_evans1975 => (key=:brown_durbin_evans1975,
        authors="Brown, Robert L. and Durbin, James and Evans, J. M.", year=1975,
        title="Techniques for Testing the Constancy of Regression Relationships over Time",
        journal="Journal of the Royal Statistical Society, Series B", volume="37", issue="2",
        pages="149--192", doi="10.1111/j.2517-6161.1975.tb01532.x", isbn="", publisher="",
        entry_type=:article),
    :chow1960 => (key=:chow1960, authors="Chow, Gregory C.", year=1960,
        title="Tests of Equality Between Sets of Coefficients in Two Linear Regressions",
        journal="Econometrica", volume="28", issue="3", pages="591--605",
        doi="10.2307/1910133", isbn="", publisher="", entry_type=:article),
    :edgerton_wells1994 => (key=:edgerton_wells1994,
        authors="Edgerton, David and Wells, Curt", year=1994,
        title="Critical Values for the CUSUMSQ Statistic in Medium and Large Sized Samples",
        journal="Oxford Bulletin of Economics and Statistics", volume="56", issue="3",
        pages="355--365", doi="10.1111/j.1468-0084.1994.mp56003008.x", isbn="", publisher="",
        entry_type=:article),
    :belsley_kuh_welsch1980 => (key=:belsley_kuh_welsch1980,
        authors="Belsley, David A. and Kuh, Edwin and Welsch, Roy E.", year=1980,
        title="Regression Diagnostics: Identifying Influential Data and Sources of Collinearity",
        journal="", volume="", issue="", pages="", doi="10.1002/0471725153",
        isbn="978-0471058564", publisher="Wiley", entry_type=:book),
    :cook1977 => (key=:cook1977, authors="Cook, R. Dennis", year=1977,
        title="Detection of Influential Observation in Linear Regression",
        journal="Technometrics", volume="19", issue="1", pages="15--18",
        doi="10.2307/1268249", isbn="", publisher="", entry_type=:article),
    # --- Long-Run Variance Toolkit (EV-12) ---
    :andrews1991 => (key=:andrews1991, authors="Andrews, Donald W. K.", year=1991,
        title="Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation",
        journal="Econometrica", volume="59", issue="3", pages="817--858",
        doi="10.2307/2938229", isbn="", publisher="", entry_type=:article),
    :newey_west1994 => (key=:newey_west1994,
        authors="Newey, Whitney K. and West, Kenneth D.", year=1994,
        title="Automatic Lag Selection in Covariance Matrix Estimation",
        journal="Review of Economic Studies", volume="61", issue="4", pages="631--653",
        doi="10.2307/2297912", isbn="", publisher="", entry_type=:article),
    :andrews_monahan1992 => (key=:andrews_monahan1992,
        authors="Andrews, Donald W. K. and Monahan, J. Christopher", year=1992,
        title="An Improved Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimator",
        journal="Econometrica", volume="60", issue="4", pages="953--966",
        doi="10.2307/2951574", isbn="", publisher="", entry_type=:article),
    :den_haan_levin1997 => (key=:den_haan_levin1997,
        authors="Den Haan, Wouter J. and Levin, Andrew T.", year=1997,
        title="A Practitioner's Guide to Robust Covariance Matrix Estimation",
        journal="Handbook of Statistics", volume="15", issue="", pages="299--342",
        doi="10.1016/S0169-7161(97)15014-3", isbn="", publisher="Elsevier", entry_type=:incollection),
    # --- Volatility Models ---
    :engle1982 => (key=:engle1982, authors="Engle, Robert F.", year=1982,
        title="Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation",
        journal="Econometrica", volume="50", issue="4", pages="987--1007",
        doi="10.2307/1912773", isbn="", publisher="", entry_type=:article),
    :bollerslev1986 => (key=:bollerslev1986, authors="Bollerslev, Tim", year=1986,
        title="Generalized Autoregressive Conditional Heteroskedasticity",
        journal="Journal of Econometrics", volume="31", issue="3", pages="307--327",
        doi="10.1016/0304-4076(86)90063-1", isbn="", publisher="", entry_type=:article),
    :nelson1991 => (key=:nelson1991, authors="Nelson, Daniel B.", year=1991,
        title="Conditional Heteroskedasticity in Asset Returns: A New Approach",
        journal="Econometrica", volume="59", issue="2", pages="347--370",
        doi="10.2307/2938260", isbn="", publisher="", entry_type=:article),
    # --- GARCH-MIDAS (EV-02, #410) ---
    :engle_ghysels_sohn2013 => (key=:engle_ghysels_sohn2013,
        authors="Engle, Robert F. and Ghysels, Eric and Sohn, Bumjean", year=2013,
        title="Stock Market Volatility and Macroeconomic Fundamentals",
        journal="Review of Economics and Statistics", volume="95", issue="3", pages="776--797",
        doi="10.1162/REST_a_00300", isbn="", publisher="", entry_type=:article),
    # --- Fractionally-integrated volatility (EV-14, #422) ---
    :baillie_bollerslev_mikkelsen1996 => (key=:baillie_bollerslev_mikkelsen1996,
        authors="Baillie, Richard T. and Bollerslev, Tim and Mikkelsen, Hans Ole", year=1996,
        title="Fractionally Integrated Generalized Autoregressive Conditional Heteroskedasticity",
        journal="Journal of Econometrics", volume="74", issue="1", pages="3--30",
        doi="10.1016/S0304-4076(95)01749-6", isbn="", publisher="", entry_type=:article),
    :bollerslev_mikkelsen1996 => (key=:bollerslev_mikkelsen1996,
        authors="Bollerslev, Tim and Mikkelsen, Hans Ole", year=1996,
        title="Modeling and Pricing Long Memory in Stock Market Volatility",
        journal="Journal of Econometrics", volume="73", issue="1", pages="151--184",
        doi="10.1016/0304-4076(95)01736-4", isbn="", publisher="", entry_type=:article),
    # --- Multivariate GARCH (EV-16, #424) ---
    :bollerslev1990 => (key=:bollerslev1990, authors="Bollerslev, Tim", year=1990,
        title="Modelling the Coherence in Short-Run Nominal Exchange Rates: A Multivariate Generalized ARCH Model",
        journal="Review of Economics and Statistics", volume="72", issue="3", pages="498--505",
        doi="10.2307/2109358", isbn="", publisher="", entry_type=:article),
    :engle2002dcc => (key=:engle2002dcc, authors="Engle, Robert F.", year=2002,
        title="Dynamic Conditional Correlation: A Simple Class of Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models",
        journal="Journal of Business \\& Economic Statistics", volume="20", issue="3", pages="339--350",
        doi="10.1198/073500102288618487", isbn="", publisher="", entry_type=:article),
    :aielli2013 => (key=:aielli2013, authors="Aielli, Gian Piero", year=2013,
        title="Dynamic Conditional Correlation: On Properties and Estimation",
        journal="Journal of Business \\& Economic Statistics", volume="31", issue="3", pages="282--299",
        doi="10.1080/07350015.2013.771027", isbn="", publisher="", entry_type=:article),
    :engle_kroner1995 => (key=:engle_kroner1995, authors="Engle, Robert F. and Kroner, Kenneth F.", year=1995,
        title="Multivariate Simultaneous Generalized ARCH",
        journal="Econometric Theory", volume="11", issue="1", pages="122--150",
        doi="10.1017/S0266466600009063", isbn="", publisher="", entry_type=:article),
    :glosten_jagannathan_runkle1993 => (key=:glosten_jagannathan_runkle1993,
        authors="Glosten, Lawrence R. and Jagannathan, Ravi and Runkle, David E.", year=1993,
        title="On the Relation Between the Expected Value and the Volatility of the Nominal Excess Return on Stocks",
        journal="Journal of Finance", volume="48", issue="5", pages="1779--1801",
        doi="10.1111/j.1540-6261.1993.tb05128.x", isbn="", publisher="", entry_type=:article),
    # --- IGARCH / Component-GARCH / APARCH + diagnostics (EV-15, #423) ---
    :engle_bollerslev1986 => (key=:engle_bollerslev1986,
        authors="Engle, Robert F. and Bollerslev, Tim", year=1986,
        title="Modelling the Persistence of Conditional Variances",
        journal="Econometric Reviews", volume="5", issue="1", pages="1--50",
        doi="10.1080/07474938608800095", isbn="", publisher="", entry_type=:article),
    :engle_lee1999 => (key=:engle_lee1999,
        authors="Engle, Robert F. and Lee, Gary G. J.", year=1999,
        title="A Permanent and Transitory Component Model of Stock Return Volatility",
        journal="", volume="", issue="", pages="475--497", doi="",
        isbn="978-0-19-510944-7",
        publisher="Oxford University Press (in Cointegration, Causality, and Forecasting, Engle & White, eds.)",
        entry_type=:incollection),
    :ding_granger_engle1993 => (key=:ding_granger_engle1993,
        authors="Ding, Zhuanxin and Granger, Clive W. J. and Engle, Robert F.", year=1993,
        title="A Long Memory Property of Stock Market Returns and a New Model",
        journal="Journal of Empirical Finance", volume="1", issue="1", pages="83--106",
        doi="10.1016/0927-5398(93)90006-D", isbn="", publisher="", entry_type=:article),
    :engle_ng1993 => (key=:engle_ng1993,
        authors="Engle, Robert F. and Ng, Victor K.", year=1993,
        title="Measuring and Testing the Impact of News on Volatility",
        journal="Journal of Finance", volume="48", issue="5", pages="1749--1778",
        doi="10.1111/j.1540-6261.1993.tb05127.x", isbn="", publisher="", entry_type=:article),
    :nyblom1989 => (key=:nyblom1989, authors="Nyblom, Jukka", year=1989,
        title="Testing for the Constancy of Parameters Over Time",
        journal="Journal of the American Statistical Association", volume="84", issue="405",
        pages="223--230", doi="10.1080/01621459.1989.10478759", isbn="", publisher="",
        entry_type=:article),
    :taylor1986 => (key=:taylor1986, authors="Taylor, Stephen J.", year=1986,
        title="Modelling Financial Time Series", journal="",
        volume="", issue="", pages="", doi="",
        isbn="978-0-471-90993-4", publisher="Wiley", entry_type=:book),
    :kim_shephard_chib1998 => (key=:kim_shephard_chib1998,
        authors="Kim, Sangjoon and Shephard, Neil and Chib, Siddhartha", year=1998,
        title="Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models",
        journal="Review of Economic Studies", volume="65", issue="3", pages="361--393",
        doi="10.1111/1467-937X.00050", isbn="", publisher="", entry_type=:article),
    :omori2007 => (key=:omori2007,
        authors="Omori, Yasuhiro and Chib, Siddhartha and Shephard, Neil and Nakajima, Jouchi", year=2007,
        title="Stochastic Volatility with Leverage: Fast and Efficient Likelihood Inference",
        journal="Journal of Econometrics", volume="140", issue="2", pages="425--449",
        doi="10.1016/j.jeconom.2006.07.008", isbn="", publisher="", entry_type=:article),
    # --- Nonlinear Time Series (threshold/SETAR) ---
    :tong1990 => (key=:tong1990, authors="Tong, Howell", year=1990,
        title="Non-linear Time Series: A Dynamical System Approach", journal="",
        volume="", issue="", pages="", doi="",
        isbn="978-0-19-852300-6", publisher="Oxford University Press", entry_type=:book),
    :hansen1996 => (key=:hansen1996, authors="Hansen, Bruce E.", year=1996,
        title="Inference When a Nuisance Parameter Is Not Identified Under the Null Hypothesis",
        journal="Econometrica", volume="64", issue="2", pages="413--430",
        doi="10.2307/2171789", isbn="", publisher="", entry_type=:article),
    :hansen2000 => (key=:hansen2000, authors="Hansen, Bruce E.", year=2000,
        title="Sample Splitting and Threshold Estimation",
        journal="Econometrica", volume="68", issue="3", pages="575--603",
        doi="10.1111/1468-0262.00124", isbn="", publisher="", entry_type=:article),
    # --- Smooth-transition autoregression (STAR) — EV-06 ---
    :luukkonen1988 => (key=:luukkonen1988,
        authors="Luukkonen, Ritva and Saikkonen, Pentti and Teräsvirta, Timo", year=1988,
        title="Testing Linearity Against Smooth Transition Autoregressive Models",
        journal="Biometrika", volume="75", issue="3", pages="491--499",
        doi="10.1093/biomet/75.3.491", isbn="", publisher="", entry_type=:article),
    :terasvirta1994 => (key=:terasvirta1994, authors="Teräsvirta, Timo", year=1994,
        title="Specification, Estimation, and Evaluation of Smooth Transition Autoregressive Models",
        journal="Journal of the American Statistical Association", volume="89", issue="425",
        pages="208--218", doi="10.1080/01621459.1994.10476462", isbn="", publisher="",
        entry_type=:article),
    # --- Markov-switching regression / MS-AR — EV-07 ---
    :hamilton1989 => (key=:hamilton1989, authors="Hamilton, James D.", year=1989,
        title="A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle",
        journal="Econometrica", volume="57", issue="2", pages="357--384",
        doi="10.2307/1912559", isbn="", publisher="", entry_type=:article),
    :kim1994 => (key=:kim1994, authors="Kim, Chang-Jin", year=1994,
        title="Dynamic Linear Models with Markov-Switching",
        journal="Journal of Econometrics", volume="60", issue="1--2", pages="1--22",
        doi="10.1016/0304-4076(94)90036-1", isbn="", publisher="", entry_type=:article),
    :giannone_lenza_primiceri2015 => (key=:giannone_lenza_primiceri2015,
        authors="Giannone, Domenico and Lenza, Michele and Primiceri, Giorgio E.", year=2015,
        title="Prior Selection for Vector Autoregressions",
        journal="Review of Economics and Statistics", volume="97", issue="2", pages="436--451",
        doi="10.1162/REST_a_00483", isbn="", publisher="", entry_type=:article),
    # --- Time Series Filters ---
    :hodrick_prescott1997 => (key=:hodrick_prescott1997,
        authors="Hodrick, Robert J. and Prescott, Edward C.", year=1997,
        title="Postwar U.S. Business Cycles: An Empirical Investigation",
        journal="Journal of Money, Credit and Banking", volume="29", issue="1", pages="1--16",
        doi="10.2307/2953682", isbn="", publisher="", entry_type=:article),
    :hamilton2018filter => (key=:hamilton2018filter,
        authors="Hamilton, James D.", year=2018,
        title="Why You Should Never Use the Hodrick-Prescott Filter",
        journal="Review of Economics and Statistics", volume="100", issue="5", pages="831--843",
        doi="10.1162/rest_a_00706", isbn="", publisher="", entry_type=:article),
    :beveridge_nelson1981 => (key=:beveridge_nelson1981,
        authors="Beveridge, Stephen and Nelson, Charles R.", year=1981,
        title="A New Approach to Decomposition of Economic Time Series into Permanent and Transitory Components with Particular Attention to Measurement of the `Business Cycle'",
        journal="Journal of Monetary Economics", volume="7", issue="2", pages="151--174",
        doi="10.1016/0304-3932(81)90040-4", isbn="", publisher="", entry_type=:article),
    :baxter_king1999 => (key=:baxter_king1999,
        authors="Baxter, Marianne and King, Robert G.", year=1999,
        title="Measuring Business Cycles: Approximate Band-Pass Filters for Economic Time Series",
        journal="Review of Economics and Statistics", volume="81", issue="4", pages="575--593",
        doi="10.1162/003465399558454", isbn="", publisher="", entry_type=:article),
    :phillips_shi2021 => (key=:phillips_shi2021,
        authors="Phillips, Peter C. B. and Shi, Zhentao", year=2021,
        title="Boosting: Why You Can Use the HP Filter",
        journal="International Economic Review", volume="62", issue="2", pages="521--570",
        doi="10.1111/iere.12495", isbn="", publisher="", entry_type=:article),
    :mei_phillips_shi2024 => (key=:mei_phillips_shi2024,
        authors="Mei, Ziwei and Phillips, Peter C. B. and Shi, Zhentao", year=2024,
        title="The boosted Hodrick-Prescott filter is more general than you might think",
        journal="Journal of Applied Econometrics", volume="39", issue="7", pages="1260--1281",
        doi="10.1002/jae.3086", isbn="", publisher="", entry_type=:article),
    # --- X-13ARIMA-SEATS ---
    :dagum_bianconcini2016 => (key=:dagum_bianconcini2016,
        authors="Dagum, Estela Bee and Bianconcini, Silvia", year=2016,
        title="Seasonal Adjustment Methods and Real Time Trend-Cycle Estimation",
        journal="", volume="", issue="", pages="",
        doi="10.1007/978-3-319-31822-6", isbn="978-3-319-31820-2", publisher="Springer", entry_type=:book),
    :findley1998 => (key=:findley1998,
        authors="Findley, David F. and Monsell, Brian C. and Bell, William R. and Otto, Mark C. and Chen, Bor-Chung", year=1998,
        title="New Capabilities and Methods of the X-12-ARIMA Seasonal-Adjustment Program",
        journal="Journal of Business and Economic Statistics", volume="16", issue="2", pages="127--152",
        doi="10.1080/07350015.1998.10524743", isbn="", publisher="", entry_type=:article),
    :gomez_maravall1996 => (key=:gomez_maravall1996,
        authors="Gómez, Víctor and Maravall, Agustín", year=1996,
        title="Programs TRAMO and SEATS: Instructions for the User",
        journal="Banco de España Working Papers", volume="9628", issue="", pages="",
        doi="", isbn="", publisher="Banco de España", entry_type=:techreport),
    # --- Model Comparison Tests ---
    :wilks1938 => (key=:wilks1938, authors="Wilks, Samuel S.", year=1938,
        title="The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses",
        journal="Annals of Mathematical Statistics", volume="9", issue="1", pages="60--62",
        doi="10.1214/aoms/1177732360", isbn="", publisher="", entry_type=:article),
    :neyman_pearson1933 => (key=:neyman_pearson1933, authors="Neyman, Jerzy and Pearson, Egon S.", year=1933,
        title="On the Problem of the Most Efficient Tests of Statistical Hypotheses",
        journal="Philosophical Transactions of the Royal Society A", volume="231", issue="694--706", pages="289--337",
        doi="10.1098/rsta.1933.0009", isbn="", publisher="", entry_type=:article),
    :rao1948 => (key=:rao1948, authors="Rao, C. Radhakrishna", year=1948,
        title="Large Sample Tests of Statistical Hypotheses Concerning Several Parameters with Applications to Problems of Estimation",
        journal="Mathematical Proceedings of the Cambridge Philosophical Society", volume="44", issue="1", pages="50--57",
        doi="10.1017/S0305004100023987", isbn="", publisher="", entry_type=:article),
    :silvey1959 => (key=:silvey1959, authors="Silvey, S. D.", year=1959,
        title="The Lagrangian Multiplier Test",
        journal="Annals of Mathematical Statistics", volume="30", issue="2", pages="389--407",
        doi="10.1214/aoms/1177706259", isbn="", publisher="", entry_type=:article),
    # --- Granger Causality ---
    :granger1969 => (key=:granger1969, authors="Granger, C. W. J.", year=1969,
        title="Investigating Causal Relations by Econometric Models and Cross-spectral Methods",
        journal="Econometrica", volume="37", issue="3", pages="424--438",
        doi="10.2307/1912791", isbn="", publisher="", entry_type=:article),
    # --- Panel VAR ---
    :holtz_eakin1988 => (key=:holtz_eakin1988,
        authors="Holtz-Eakin, Douglas and Newey, Whitney and Rosen, Harvey S.",
        year=1988, title="Estimating Vector Autoregressions with Panel Data",
        journal="Econometrica", volume="56", issue="6", pages="1371--1395",
        doi="10.2307/1913103", isbn="", publisher="", entry_type=:article),
    :arellano_bond1991 => (key=:arellano_bond1991,
        authors="Arellano, Manuel and Bond, Stephen", year=1991,
        title="Some Tests of Specification for Panel Data: Monte Carlo Evidence and an Application to Employment Equations",
        journal="Review of Economic Studies", volume="58", issue="2", pages="277--297",
        doi="10.2307/2297968", isbn="", publisher="", entry_type=:article),
    :blundell_bond1998 => (key=:blundell_bond1998,
        authors="Blundell, Richard and Bond, Stephen", year=1998,
        title="Initial Conditions and Moment Restrictions in Dynamic Panel Data Models",
        journal="Journal of Econometrics", volume="87", issue="1", pages="115--143",
        doi="10.1016/S0304-4076(98)00009-8", isbn="", publisher="", entry_type=:article),
    :windmeijer2005 => (key=:windmeijer2005,
        authors="Windmeijer, Frank", year=2005,
        title="A Finite Sample Correction for the Variance of Linear Efficient Two-Step GMM Estimators",
        journal="Journal of Econometrics", volume="126", issue="1", pages="25--51",
        doi="10.1016/j.jeconom.2004.02.005", isbn="", publisher="", entry_type=:article),
    :andrews_lu2001 => (key=:andrews_lu2001,
        authors="Andrews, Donald W. K. and Lu, Biao", year=2001,
        title="Consistent Model and Moment Selection Procedures for GMM Estimation with Application to Dynamic Panel Data Models",
        journal="Journal of Econometrics", volume="101", issue="1", pages="123--164",
        doi="10.1016/S0304-4076(00)00077-4", isbn="", publisher="", entry_type=:article),
    :pesaran_shin1998 => (key=:pesaran_shin1998,
        authors="Pesaran, M. Hashem and Shin, Yongcheol", year=1998,
        title="Generalized Impulse Response Analysis in Linear Multivariate Models",
        journal="Economics Letters", volume="58", issue="1", pages="17--29",
        doi="10.1016/S0165-1765(97)00214-0", isbn="", publisher="", entry_type=:article),
    :binder_hsiao_pesaran2005 => (key=:binder_hsiao_pesaran2005,
        authors="Binder, Michael and Hsiao, Cheng and Pesaran, M. Hashem", year=2005,
        title="Estimation and Inference in Short Panel Vector Autoregressions with Unit Roots and Cointegration",
        journal="Econometric Theory", volume="21", issue="4", pages="795--837",
        doi="10.1017/S0266466605050413", isbn="", publisher="", entry_type=:article),
    # --- Data Sources ---
    :mccracken_ng2016 => (key=:mccracken_ng2016,
        authors="McCracken, Michael W. and Ng, Serena", year=2016,
        title="FRED-MD: A Monthly Database for Macroeconomic Research",
        journal="Journal of Business \\& Economic Statistics", volume="34", issue="4", pages="574--589",
        doi="10.1080/07350015.2015.1086655", isbn="", publisher="", entry_type=:article),
    :mccracken_ng2020 => (key=:mccracken_ng2020,
        authors="McCracken, Michael W. and Ng, Serena", year=2020,
        title="FRED-QD: A Quarterly Database for Macroeconomic Research",
        journal="Federal Reserve Bank of St. Louis Working Paper", volume="2020-005", issue="", pages="",
        doi="10.20955/wp.2020.005", isbn="", publisher="", entry_type=:article),
    :feenstra_etal2015 => (key=:feenstra_etal2015,
        authors="Feenstra, Robert C. and Inklaar, Robert and Timmer, Marcel P.", year=2015,
        title="The Next Generation of the Penn World Table",
        journal="American Economic Review", volume="105", issue="10", pages="3150--3182",
        doi="10.1257/aer.20130954", isbn="", publisher="", entry_type=:article),
    # --- Nowcasting ---
    :banbura_modugno2014 => (key=:banbura_modugno2014,
        authors="Ba{\\'n}bura, Marta and Modugno, Michele", year=2014,
        title="Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data",
        journal="Journal of Applied Econometrics", volume="29", issue="1", pages="133--160",
        doi="10.1002/jae.2306", isbn="", publisher="", entry_type=:article),
    :delle_chiaie2022 => (key=:delle_chiaie2022,
        authors="Delle Chiaie, Simona and Ferrara, Laurent and Giannone, Domenico", year=2022,
        title="Common Factors of Commodity Prices",
        journal="Journal of Applied Econometrics", volume="37", issue="3", pages="461--476",
        doi="10.1002/jae.2887", isbn="", publisher="", entry_type=:article),
    :cimadomo2022 => (key=:cimadomo2022,
        authors="Cimadomo, Jacopo and Giannone, Domenico and Lenza, Michele and Monti, Francesca and Sokol, Andrej", year=2022,
        title="Nowcasting with Large Bayesian Vector Autoregressions",
        journal="Journal of Econometrics", volume="231", issue="2", pages="500--519",
        doi="10.1016/j.jeconom.2021.04.012", isbn="", publisher="", entry_type=:article),
    :banbura2023 => (key=:banbura2023,
        authors="Ba{\\'n}bura, Marta and Belousova, Irina and Bodn\\'ar, Katalin and T\\'oth, M\\'at\\'e Barnab\\'as", year=2023,
        title="Nowcasting Employment in the Euro Area",
        journal="Working Paper Series", volume="No 2815", issue="", pages="",
        doi="", isbn="", publisher="European Central Bank", entry_type=:techreport),
    # --- Sign Restriction Identified Set ---
    :baumeister_hamilton2015 => (key=:baumeister_hamilton2015,
        authors="Baumeister, Christiane and Hamilton, James D.", year=2015,
        title="Sign Restrictions, Structural Vector Autoregressions, and Useful Prior Information",
        journal="Econometrica", volume="83", issue="5", pages="1963--1999",
        doi="10.3982/ECTA12356", isbn="", publisher="", entry_type=:article),
    :rubio_ramirez2010 => (key=:rubio_ramirez2010,
        authors="Rubio-Ram{\\'\\i}rez, Juan F. and Waggoner, Daniel F. and Zha, Tao", year=2010,
        title="Structural Vector Autoregressions: Theory of Identification and Algorithms for Inference",
        journal="Review of Economic Studies", volume="77", issue="2", pages="665--696",
        doi="10.1111/j.1467-937X.2009.00578.x", isbn="", publisher="", entry_type=:article),
    # --- Morley-Nelson-Zivot UC Model ---
    :morley_nelson_zivot2003 => (key=:morley_nelson_zivot2003,
        authors="Morley, James C. and Nelson, Charles R. and Zivot, Eric", year=2003,
        title="Why Are the Beveridge-Nelson and Unobserved-Components Decompositions of GDP So Different?",
        journal="Review of Economics and Statistics", volume="85", issue="2", pages="235--243",
        doi="10.1162/003465303765299882", isbn="", publisher="", entry_type=:article),
    # --- DSGE ---
    :sims2002 => (key=:sims2002, authors="Sims, Christopher A.", year=2002,
        title="Solving Linear Rational Expectations Models",
        journal="Computational Economics", volume="20", issue="1--2", pages="1--20",
        doi="10.1023/A:1020517101123", isbn="", publisher="", entry_type=:article),
    :blanchard_kahn1980 => (key=:blanchard_kahn1980,
        authors="Blanchard, Olivier Jean and Kahn, Charles M.", year=1980,
        title="The Solution of Linear Difference Models Under Rational Expectations",
        journal="Econometrica", volume="48", issue="5", pages="1305--1311",
        doi="10.2307/1912186", isbn="", publisher="", entry_type=:article),
    :christiano_eichenbaum_evans2005 => (key=:christiano_eichenbaum_evans2005,
        authors="Christiano, Lawrence J. and Eichenbaum, Martin and Evans, Charles L.", year=2005,
        title="Nominal Rigidities and the Dynamic Effects of a Shock to Monetary Policy",
        journal="Journal of Political Economy", volume="113", issue="1", pages="1--45",
        doi="10.1086/426038", isbn="", publisher="", entry_type=:article),
    :hansen_singleton1982 => (key=:hansen_singleton1982,
        authors="Hansen, Lars Peter and Singleton, Kenneth J.", year=1982,
        title="Generalized Instrumental Variables Estimation of Nonlinear Rational Expectations Models",
        journal="Econometrica", volume="50", issue="5", pages="1269--1286",
        doi="10.2307/1911873", isbn="", publisher="", entry_type=:article),
    # --- OccBin ---
    :guerrieri_iacoviello2015 => (key=:guerrieri_iacoviello2015,
        authors="Guerrieri, Luca and Iacoviello, Matteo", year=2015,
        title="OccBin: A Toolkit for Solving Dynamic Models with Occasionally Binding Constraints Easily",
        journal="Journal of Monetary Economics", volume="70", issue="", pages="22--38",
        doi="10.1016/j.jmoneco.2014.08.005", isbn="", publisher="", entry_type=:article),
    # --- DSGE analytical ---
    :fernandez_villaverde_rubio_schorfheide2016 => (
        key=:fernandez_villaverde_rubio_schorfheide2016,
        authors="Fernández-Villaverde, Jesús and Rubio-Ramírez, Juan F. and Schorfheide, Frank",
        year=2016,
        title="Solution and Estimation Methods for DSGE Models",
        journal="Handbook of Macroeconomics", volume="2", issue="", pages="527--724",
        doi="10.1016/bs.hesmac.2016.03.006", isbn="", publisher="Elsevier",
        entry_type=:incollection),
    # --- DSGE solvers ---
    :klein2000 => (key=:klein2000, authors="Klein, Paul", year=2000,
        title="Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model",
        journal="Journal of Economic Dynamics and Control", volume="24", issue="10", pages="1405--1423",
        doi="10.1016/S0165-1889(99)00045-7", isbn="", publisher="", entry_type=:article),
    :schmitt_grohe_uribe2004 => (key=:schmitt_grohe_uribe2004,
        authors="Schmitt-Groh\u00e9, Stephanie and Uribe, Mart\u00edn", year=2004,
        title="Solving Dynamic General Equilibrium Models Using a Second-Order Approximation to the Policy Function",
        journal="Journal of Economic Dynamics and Control", volume="28", issue="4", pages="755--775",
        doi="10.1016/S0165-1889(03)00043-5", isbn="", publisher="", entry_type=:article),
    :kim_kim_schaumburg_sims2008 => (key=:kim_kim_schaumburg_sims2008,
        authors="Kim, Jinill and Kim, Sunghyun Henry and Schaumburg, Ernst and Sims, Christopher A.", year=2008,
        title="Calculating and Using Second-Order Accurate Solutions of Discrete Time Dynamic Equilibrium Models",
        journal="Journal of Economic Dynamics and Control", volume="32", issue="11", pages="3397--3414",
        doi="10.1016/j.jedc.2008.02.003", isbn="", publisher="", entry_type=:article),
    :andreasen_etal2018 => (key=:andreasen_etal2018,
        authors="Andreasen, Martin M. and Fern\u00e1ndez-Villaverde, Jes\u00fas and Rubio-Ram\u00edrez, Juan F.",
        year=2018,
        title="The Pruned State-Space System for Non-Linear DSGE Models: Theory and Empirical Applications",
        journal="Review of Economic Studies", volume="85", issue="1", pages="1--49",
        doi="10.1093/restud/rdx037", isbn="", publisher="", entry_type=:article),
    # --- DSGE projection/PFI ---
    :coleman1990 => (key=:coleman1990, authors="Coleman, Wilbur John, II", year=1990,
        title="Solving the Stochastic Growth Model by Policy-Function Iteration",
        journal="Journal of Business \\& Economic Statistics", volume="8", issue="1", pages="27--29",
        doi="10.1080/07350015.1990.10509769", isbn="", publisher="", entry_type=:article),
    :judd1992 => (key=:judd1992, authors="Judd, Kenneth L.", year=1992,
        title="Projection Methods for Solving Aggregate Growth Models",
        journal="Journal of Economic Theory", volume="58", issue="2", pages="410--452",
        doi="10.1016/0022-0531(92)90061-L", isbn="", publisher="", entry_type=:article),
    :judd1998 => (key=:judd1998, authors="Judd, Kenneth L.", year=1998,
        title="Numerical Methods in Economics",
        journal="", volume="", issue="", pages="",
        doi="", isbn="0-262-10071-1", publisher="MIT Press", entry_type=:book),
    :judd_maliar_maliar_valero2014 => (key=:judd_maliar_maliar_valero2014,
        authors="Judd, Kenneth L. and Maliar, Lilia and Maliar, Serguei and Valero, Rafael", year=2014,
        title="Smolyak Method for Solving Dynamic Economic Models: Lagrange Interpolation, Anisotropic Grid and Adaptive Domain",
        journal="Journal of Economic Dynamics and Control", volume="44", issue="", pages="92--123",
        doi="10.1016/j.jedc.2014.03.003", isbn="", publisher="", entry_type=:article),
    # --- DSGE VFI / Anderson ---
    :stokey_lucas_prescott1989 => (key=:stokey_lucas_prescott1989,
        authors="Stokey, Nancy L. and Lucas, Robert E. and Prescott, Edward C.", year=1989,
        title="Recursive Methods in Economic Dynamics",
        journal="", volume="", issue="", pages="",
        doi="", isbn="0-674-75096-9", publisher="Harvard University Press", entry_type=:book),
    :howard1960 => (key=:howard1960,
        authors="Howard, Ronald A.", year=1960,
        title="Dynamic Programming and Markov Processes",
        journal="", volume="", issue="", pages="",
        doi="", isbn="", publisher="MIT Press", entry_type=:book),
    :santos_rust2003 => (key=:santos_rust2003,
        authors="Santos, Manuel S. and Rust, John", year=2003,
        title="Convergence Properties of Policy Iteration",
        journal="SIAM Journal on Control and Optimization", volume="42", issue="6", pages="2094--2115",
        doi="10.1137/S0363012902399824", isbn="", publisher="", entry_type=:article),
    :walker_ni2011 => (key=:walker_ni2011,
        authors="Walker, Homer F. and Ni, Peng", year=2011,
        title="Anderson Acceleration for Fixed-Point Iterations",
        journal="SIAM Journal on Numerical Analysis", volume="49", issue="4", pages="1715--1735",
        doi="10.1137/10078356X", isbn="", publisher="", entry_type=:article),
    # --- DSGE GIRF ---
    :koop_pesaran_potter1996 => (key=:koop_pesaran_potter1996,
        authors="Koop, Gary and Pesaran, M. Hashem and Potter, Simon M.", year=1996,
        title="Impulse Response Analysis in Nonlinear Multivariate Models",
        journal="Journal of Econometrics", volume="74", issue="1", pages="119--147",
        doi="10.1016/0304-4076(95)01753-4", isbn="", publisher="", entry_type=:article),
    # --- DSGE estimation ---
    :smets_wouters2007 => (key=:smets_wouters2007,
        authors="Smets, Frank and Wouters, Rafael", year=2007,
        title="Shocks and Frictions in US Business Cycles: A Bayesian DSGE Approach",
        journal="American Economic Review", volume="97", issue="3", pages="586--606",
        doi="10.1257/aer.97.3.586", isbn="", publisher="", entry_type=:article),
    # --- Difference-in-Differences ---
    :callaway_santanna2021 => (key=:callaway_santanna2021,
        authors="Callaway, Brantly and Sant'Anna, Pedro H. C.", year=2021,
        title="Difference-in-Differences with Multiple Time Periods",
        journal="Journal of Econometrics", volume="225", issue="2", pages="200--230",
        doi="10.1016/j.jeconom.2020.12.001", isbn="", publisher="", entry_type=:article),
    :goodman_bacon2021 => (key=:goodman_bacon2021,
        authors="Goodman-Bacon, Andrew", year=2021,
        title="Difference-in-Differences with Variation in Treatment Timing",
        journal="Journal of Econometrics", volume="225", issue="2", pages="254--277",
        doi="10.1016/j.jeconom.2021.03.014", isbn="", publisher="", entry_type=:article),
    :dechaisemartin_dhaultfoeuille2020 => (key=:dechaisemartin_dhaultfoeuille2020,
        authors="de Chaisemartin, Cl\u00e9ment and D'Haultf\u0153uille, Xavier", year=2020,
        title="Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects",
        journal="American Economic Review", volume="110", issue="9", pages="2964--2996",
        doi="10.1257/aer.20181169", isbn="", publisher="", entry_type=:article),
    :dube_girardi_jorda_taylor2023 => (key=:dube_girardi_jorda_taylor2023,
        authors="Dube, Arindrajit and Girardi, Daniele and Jord\u00e0, \u00d2scar and Taylor, Alan M.",
        year=2025, title="A Local Projections Approach to Difference-in-Differences",
        journal="Journal of Applied Econometrics", volume="", issue="", pages="",
        doi="10.1002/jae.3117", isbn="", publisher="", entry_type=:article),
    :sun_abraham2021 => (key=:sun_abraham2021,
        authors="Sun, Liyang and Abraham, Sarah", year=2021,
        title="Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects",
        journal="Journal of Econometrics", volume="225", issue="2", pages="175--199",
        doi="10.1016/j.jeconom.2020.09.006", isbn="", publisher="", entry_type=:article),
    :borusyak_jaravel_spiess2024 => (key=:borusyak_jaravel_spiess2024,
        authors="Borusyak, Kirill and Jaravel, Xavier and Spiess, Jann", year=2024,
        title="Revisiting Event-Study Designs: Robust and Efficient Estimation",
        journal="Review of Economic Studies", volume="91", issue="6", pages="3253--3285",
        doi="10.1093/restud/rdae007", isbn="", publisher="", entry_type=:article),
    :rambachan_roth2023 => (key=:rambachan_roth2023,
        authors="Rambachan, Ashesh and Roth, Jonathan", year=2023,
        title="A More Credible Approach to Parallel Trends",
        journal="Review of Economic Studies", volume="90", issue="5", pages="2555--2591",
        doi="10.1093/restud/rdad018", isbn="", publisher="", entry_type=:article),
    :armstrong_kolesar2018 => (key=:armstrong_kolesar2018,
        authors="Armstrong, Timothy B. and Kolesár, Michal", year=2018,
        title="Optimal Inference in a Class of Regression Models",
        journal="Econometrica", volume="86", issue="2", pages="655--683",
        doi="10.3982/ECTA14434", isbn="", publisher="", entry_type=:article),
    :cameron_gelbach_miller2011 => (key=:cameron_gelbach_miller2011,
        authors="Cameron, A. Colin and Gelbach, Jonah B. and Miller, Douglas L.", year=2011,
        title="Robust Inference with Multiway Clustering",
        journal="Journal of Business \\& Economic Statistics", volume="29", issue="2",
        pages="238--249", doi="10.1198/jbes.2010.07136", isbn="", publisher="",
        entry_type=:article),
    :herbst_schorfheide2015 => (key=:herbst_schorfheide2015,
        authors="Herbst, Edward P. and Schorfheide, Frank", year=2015,
        title="Bayesian Estimation of DSGE Models",
        journal="", volume="", issue="", pages="",
        doi="10.1515/9781400873739", isbn="978-0-691-16108-2",
        publisher="Princeton University Press", entry_type=:book),
    :herbst_schorfheide2014 => (key=:herbst_schorfheide2014,
        authors="Herbst, Edward and Schorfheide, Frank", year=2014,
        title="Sequential Monte Carlo Sampling for DSGE Models",
        journal="Journal of Applied Econometrics", volume="29", issue="7",
        pages="1073--1098", doi="10.1002/jae.2397", isbn="", publisher="",
        entry_type=:article),
    :chopin_jacob_papaspiliopoulos2013 => (key=:chopin_jacob_papaspiliopoulos2013,
        authors="Chopin, Nicolas and Jacob, Pierre E. and Papaspiliopoulos, Omiros",
        year=2013,
        title="SMC2: An Efficient Algorithm for Sequential Analysis of State Space Models",
        journal="Journal of the Royal Statistical Society: Series B",
        volume="75", issue="3", pages="397--426",
        doi="10.1111/j.1467-9868.2012.01046.x", isbn="", publisher="",
        entry_type=:article),
    :an_schorfheide2007 => (key=:an_schorfheide2007,
        authors="An, Sungbae and Schorfheide, Frank", year=2007,
        title="Bayesian Analysis of DSGE Models",
        journal="Econometric Reviews", volume="26", issue="2-4",
        pages="113--172", doi="10.1080/07474930701220071", isbn="", publisher="",
        entry_type=:article),
    :andrieu_doucet_holenstein2010 => (key=:andrieu_doucet_holenstein2010,
        authors="Andrieu, Christophe and Doucet, Arnaud and Holenstein, Roman",
        year=2010,
        title="Particle Markov Chain Monte Carlo Methods",
        journal="Journal of the Royal Statistical Society: Series B",
        volume="72", issue="3", pages="269--342",
        doi="10.1111/j.1467-9868.2009.00736.x", isbn="", publisher="",
        entry_type=:article),
    :gordon_salmond_smith1993 => (key=:gordon_salmond_smith1993,
        authors="Gordon, Neil J. and Salmond, David J. and Smith, Adrian F. M.",
        year=1993,
        title="Novel Approach to Nonlinear/Non-Gaussian Bayesian State Estimation",
        journal="IEE Proceedings F - Radar and Signal Processing",
        volume="140", issue="2", pages="107--113",
        doi="10.1049/ip-f-2.1993.0015", isbn="", publisher="",
        entry_type=:article),
    :pitt_shephard1999 => (key=:pitt_shephard1999,
        authors="Pitt, Michael K. and Shephard, Neil", year=1999,
        title="Filtering via Simulation: Auxiliary Particle Filters",
        journal="Journal of the American Statistical Association",
        volume="94", issue="446", pages="590--599",
        doi="10.1080/01621459.1999.10474153", isbn="", publisher="",
        entry_type=:article),
    # --- FAVAR & Structural DFM ---
    :bernanke_boivin_eliasz2005 => (key=:bernanke_boivin_eliasz2005,
        authors="Bernanke, Ben S. and Boivin, Jean and Eliasz, Piotr", year=2005,
        title="Measuring the Effects of Monetary Policy: A Factor-Augmented Vector Autoregressive (FAVAR) Approach",
        journal="Quarterly Journal of Economics", volume="120", issue="1", pages="387--422",
        doi="10.1162/0033553053970344", isbn="", publisher="", entry_type=:article),
    :forni_giannone_lippi_reichlin2009 => (key=:forni_giannone_lippi_reichlin2009,
        authors="Forni, Mario and Giannone, Domenico and Lippi, Marco and Reichlin, Lucrezia", year=2009,
        title="Opening the Black Box: Structural Factor Models with Large Cross Sections",
        journal="Econometric Theory", volume="25", issue="5", pages="1319--1347",
        doi="10.1017/S026646660809052X", isbn="", publisher="", entry_type=:article),
    # --- Structural Break Tests ---
    :andrews1993 => (key=:andrews1993, authors="Andrews, Donald W. K.", year=1993,
        title="Tests for Parameter Instability and Structural Change with Unknown Change Point",
        journal="Econometrica", volume="61", issue="4", pages="821--856",
        doi="10.2307/2951764", isbn="", publisher="", entry_type=:article),
    :andrews_ploberger1994 => (key=:andrews_ploberger1994,
        authors="Andrews, Donald W. K. and Ploberger, Werner", year=1994,
        title="Optimal Tests When a Nuisance Parameter Is Present Only Under the Alternative",
        journal="Econometrica", volume="62", issue="6", pages="1383--1414",
        doi="10.2307/2951753", isbn="", publisher="", entry_type=:article),
    :hansen1997 => (key=:hansen1997, authors="Hansen, Bruce E.", year=1997,
        title="Approximate Asymptotic P Values for Structural-Change Tests",
        journal="Journal of Business \\& Economic Statistics", volume="15", issue="1",
        pages="60--67", doi="10.1080/07350015.1997.10524687", isbn="", publisher="",
        entry_type=:article),
    :bai_perron1998 => (key=:bai_perron1998, authors="Bai, Jushan and Perron, Pierre", year=1998,
        title="Estimating and Testing Linear Models with Multiple Structural Changes",
        journal="Econometrica", volume="66", issue="1", pages="47--78",
        doi="10.2307/2998540", isbn="", publisher="", entry_type=:article),
    :bai_perron2003 => (key=:bai_perron2003, authors="Bai, Jushan and Perron, Pierre", year=2003,
        title="Computation and Analysis of Multiple Structural Change Models",
        journal="Journal of Applied Econometrics", volume="18", issue="1", pages="1--22",
        doi="10.1002/jae.659", isbn="", publisher="", entry_type=:article),
    # --- Panel Unit Root Tests ---
    :bai_ng2004 => (key=:bai_ng2004, authors="Bai, Jushan and Ng, Serena", year=2004,
        title="A PANIC Attack on Unit Roots and Cointegration",
        journal="Econometrica", volume="72", issue="4", pages="1127--1177",
        doi="10.1111/j.1468-0262.2004.00528.x", isbn="", publisher="", entry_type=:article),
    :bai_ng2010 => (key=:bai_ng2010, authors="Bai, Jushan and Ng, Serena", year=2010,
        title="Panel Unit Root Tests with Cross-Section Dependence: A Further Investigation",
        journal="Econometric Theory", volume="26", issue="4", pages="1088--1114",
        doi="10.1017/S0266466609990478", isbn="", publisher="", entry_type=:article),
    :pesaran2007 => (key=:pesaran2007, authors="Pesaran, M. Hashem", year=2007,
        title="A Simple Panel Unit Root Test in the Presence of Cross-Section Dependence",
        journal="Journal of Applied Econometrics", volume="22", issue="2", pages="265--312",
        doi="10.1002/jae.951", isbn="", publisher="", entry_type=:article),
    :moon_perron2004 => (key=:moon_perron2004, authors="Moon, Hyungsik Roger and Perron, Benoit", year=2004,
        title="Testing for a Unit Root in Panels with Dynamic Factors",
        journal="Journal of Econometrics", volume="122", issue="1", pages="81--126",
        doi="10.1016/j.jeconom.2003.10.020", isbn="", publisher="", entry_type=:article),
    # --- First-Generation Panel Unit Root Tests (EV-20, #428) ---
    :levin_lin_chu2002 => (key=:levin_lin_chu2002, authors="Levin, Andrew and Lin, Chien-Fu and Chu, Chia-Shang James", year=2002,
        title="Unit Root Tests in Panel Data: Asymptotic and Finite-Sample Properties",
        journal="Journal of Econometrics", volume="108", issue="1", pages="1--24",
        doi="10.1016/S0304-4076(01)00098-7", isbn="", publisher="", entry_type=:article),
    :im_pesaran_shin2003 => (key=:im_pesaran_shin2003, authors="Im, Kyung So and Pesaran, M. Hashem and Shin, Yongcheol", year=2003,
        title="Testing for Unit Roots in Heterogeneous Panels",
        journal="Journal of Econometrics", volume="115", issue="1", pages="53--74",
        doi="10.1016/S0304-4076(03)00092-7", isbn="", publisher="", entry_type=:article),
    :breitung2000 => (key=:breitung2000, authors="Breitung, Jorg", year=2000,
        title="The Local Power of Some Unit Root Tests for Panel Data",
        journal="Advances in Econometrics", volume="15", issue="", pages="161--178",
        doi="10.1016/S0731-9053(00)15006-6", isbn="", publisher="JAI Press", entry_type=:incollection),
    :maddala_wu1999 => (key=:maddala_wu1999, authors="Maddala, G. S. and Wu, Shaowen", year=1999,
        title="A Comparative Study of Unit Root Tests with Panel Data and a New Simple Test",
        journal="Oxford Bulletin of Economics and Statistics", volume="61", issue="S1", pages="631--652",
        doi="10.1111/1468-0084.61.s1.13", isbn="", publisher="", entry_type=:article),
    :choi2001 => (key=:choi2001, authors="Choi, In", year=2001,
        title="Unit Root Tests for Panel Data",
        journal="Journal of International Money and Finance", volume="20", issue="2", pages="249--272",
        doi="10.1016/S0261-5606(00)00048-6", isbn="", publisher="", entry_type=:article),
    :hadri2000 => (key=:hadri2000, authors="Hadri, Kaddour", year=2000,
        title="Testing for Stationarity in Heterogeneous Panel Data",
        journal="Econometrics Journal", volume="3", issue="2", pages="148--161",
        doi="10.1111/1368-423X.00043", isbn="", publisher="", entry_type=:article),
    # --- Panel Cointegration Tests (EV-21, #429) ---
    :pedroni1999 => (key=:pedroni1999, authors="Pedroni, Peter", year=1999,
        title="Critical Values for Cointegration Tests in Heterogeneous Panels with Multiple Regressors",
        journal="Oxford Bulletin of Economics and Statistics", volume="61", issue="S1", pages="653--670",
        doi="10.1111/1468-0084.61.s1.14", isbn="", publisher="", entry_type=:article),
    :pedroni2004 => (key=:pedroni2004, authors="Pedroni, Peter", year=2004,
        title="Panel Cointegration: Asymptotic and Finite Sample Properties of Pooled Time Series Tests with an Application to the PPP Hypothesis",
        journal="Econometric Theory", volume="20", issue="3", pages="597--625",
        doi="10.1017/S0266466604203073", isbn="", publisher="", entry_type=:article),
    :kao1999 => (key=:kao1999, authors="Kao, Chihwa", year=1999,
        title="Spurious Regression and Residual-Based Tests for Cointegration in Panel Data",
        journal="Journal of Econometrics", volume="90", issue="1", pages="1--44",
        doi="10.1016/S0304-4076(98)00023-2", isbn="", publisher="", entry_type=:article),
    :westerlund2007 => (key=:westerlund2007, authors="Westerlund, Joakim", year=2007,
        title="Testing for Error Correction in Panel Data",
        journal="Oxford Bulletin of Economics and Statistics", volume="69", issue="6", pages="709--748",
        doi="10.1111/j.1468-0084.2007.00477.x", isbn="", publisher="", entry_type=:article),
    :persyn_westerlund2008 => (key=:persyn_westerlund2008, authors="Persyn, Damiaan and Westerlund, Joakim", year=2008,
        title="Error-Correction-Based Cointegration Tests for Panel Data",
        journal="Stata Journal", volume="8", issue="2", pages="232--241",
        doi="10.1177/1536867X0800800205", isbn="", publisher="", entry_type=:article),
    # --- Panel Cointegrating Regression (EV-22, #430) ---
    :pedroni2000 => (key=:pedroni2000, authors="Pedroni, Peter", year=2000,
        title="Fully Modified OLS for Heterogeneous Cointegrated Panels",
        journal="Advances in Econometrics", volume="15", issue="", pages="93--130",
        doi="10.1016/S0731-9053(00)15004-2", isbn="", publisher="", entry_type=:article),
    :pedroni2001 => (key=:pedroni2001, authors="Pedroni, Peter", year=2001,
        title="Purchasing Power Parity Tests in Cointegrated Panels",
        journal="The Review of Economics and Statistics", volume="83", issue="4", pages="727--731",
        doi="10.1162/003465301753237803", isbn="", publisher="", entry_type=:article),
    :kao_chiang2000 => (key=:kao_chiang2000, authors="Kao, Chihwa and Chiang, Min-Hsien", year=2000,
        title="On the Estimation and Inference of a Cointegrated Regression in Panel Data",
        journal="Advances in Econometrics", volume="15", issue="", pages="179--222",
        doi="10.1016/S0731-9053(00)15007-8", isbn="", publisher="", entry_type=:article),
    :mark_sul2003 => (key=:mark_sul2003, authors="Mark, Nelson C. and Sul, Donggyu", year=2003,
        title="Cointegration Vector Estimation by Panel DOLS and Long-Run Money Demand",
        journal="Oxford Bulletin of Economics and Statistics", volume="65", issue="5", pages="655--680",
        doi="10.1111/j.1468-0084.2003.00066.x", isbn="", publisher="", entry_type=:article),
    # --- Factor Model Break Tests ---
    :breitung_eickmeier2011 => (key=:breitung_eickmeier2011,
        authors="Breitung, Jorg and Eickmeier, Sandra", year=2011,
        title="Testing for Structural Breaks in Dynamic Factor Models",
        journal="Journal of Econometrics", volume="163", issue="1", pages="71--84",
        doi="10.1016/j.jeconom.2010.11.008", isbn="", publisher="", entry_type=:article),
    :chen_dolado_gonzalo2014 => (key=:chen_dolado_gonzalo2014,
        authors="Chen, Liang and Dolado, Juan J. and Gonzalo, Jesus", year=2014,
        title="Detecting Big Structural Breaks in Large Factor Models",
        journal="Journal of Econometrics", volume="180", issue="1", pages="30--48",
        doi="10.1016/j.jeconom.2014.01.006", isbn="", publisher="", entry_type=:article),
    :han_inoue2015 => (key=:han_inoue2015, authors="Han, Xu and Inoue, Atsushi", year=2015,
        title="Tests for Parameter Instability in Dynamic Factor Models",
        journal="Econometric Theory", volume="31", issue="5", pages="1117--1152",
        doi="10.1017/S0266466614000413", isbn="", publisher="", entry_type=:article),
    # --- Cross-Sectional Models ---
    :wooldridge2010 => (key=:wooldridge2010, authors="Wooldridge, Jeffrey M.", year=2010,
        title="Econometric Analysis of Cross Section and Panel Data", journal="",
        volume="", issue="", pages="", doi="",
        isbn="978-0-262-23258-6", publisher="MIT Press", entry_type=:book),
    :cameron_trivedi2005 => (key=:cameron_trivedi2005, authors="Cameron, A. Colin and Trivedi, Pravin K.", year=2005,
        title="Microeconometrics: Methods and Applications", journal="",
        volume="", issue="", pages="", doi="",
        isbn="978-0-521-84805-3", publisher="Cambridge University Press", entry_type=:book),
    :mcfadden1974 => (key=:mcfadden1974, authors="McFadden, Daniel", year=1974,
        title="Conditional Logit Analysis of Qualitative Choice Behavior",
        journal="Frontiers in Econometrics", volume="", issue="", pages="105--142",
        doi="", isbn="", publisher="Academic Press", entry_type=:incollection),
    :staiger_stock1997 => (key=:staiger_stock1997, authors="Staiger, Douglas and Stock, James H.", year=1997,
        title="Instrumental Variables Regression with Weak Instruments",
        journal="Econometrica", volume="65", issue="3", pages="557--586",
        doi="10.2307/2171753", isbn="", publisher="", entry_type=:article),
    # --- IV k-class / LIML / Fuller estimators (EV-36, #444) ---
    :anderson_rubin1949 => (key=:anderson_rubin1949,
        authors="Anderson, T. W. and Rubin, Herman", year=1949,
        title="Estimation of the Parameters of a Single Equation in a Complete System of Stochastic Equations",
        journal="Annals of Mathematical Statistics", volume="20", issue="1", pages="46--63",
        doi="10.1214/aoms/1177730090", isbn="", publisher="", entry_type=:article),
    :fuller1977 => (key=:fuller1977, authors="Fuller, Wayne A.", year=1977,
        title="Some Properties of a Modification of the Limited Information Estimator",
        journal="Econometrica", volume="45", issue="4", pages="939--953",
        doi="10.2307/1912683", isbn="", publisher="", entry_type=:article),
    :bekker1994 => (key=:bekker1994, authors="Bekker, Paul A.", year=1994,
        title="Alternative Approximations to the Distributions of Instrumental Variable Estimators",
        journal="Econometrica", volume="62", issue="3", pages="657--681",
        doi="10.2307/2951662", isbn="", publisher="", entry_type=:article),
    # --- Penalized Regression (EV-03, #411) ---
    :hoerl_kennard1970 => (key=:hoerl_kennard1970,
        authors="Hoerl, Arthur E. and Kennard, Robert W.", year=1970,
        title="Ridge Regression: Biased Estimation for Nonorthogonal Problems",
        journal="Technometrics", volume="12", issue="1", pages="55--67",
        doi="10.1080/00401706.1970.10488634", isbn="", publisher="", entry_type=:article),
    :tibshirani1996 => (key=:tibshirani1996, authors="Tibshirani, Robert", year=1996,
        title="Regression Shrinkage and Selection via the Lasso",
        journal="Journal of the Royal Statistical Society, Series B",
        volume="58", issue="1", pages="267--288",
        doi="10.1111/j.2517-6161.1996.tb02080.x", isbn="", publisher="", entry_type=:article),
    :zou_hastie2005 => (key=:zou_hastie2005, authors="Zou, Hui and Hastie, Trevor", year=2005,
        title="Regularization and Variable Selection via the Elastic Net",
        journal="Journal of the Royal Statistical Society, Series B",
        volume="67", issue="2", pages="301--320",
        doi="10.1111/j.1467-9868.2005.00503.x", isbn="", publisher="", entry_type=:article),
    :zou2006 => (key=:zou2006, authors="Zou, Hui", year=2006,
        title="The Adaptive Lasso and Its Oracle Properties",
        journal="Journal of the American Statistical Association",
        volume="101", issue="476", pages="1418--1429",
        doi="10.1198/016214506000000735", isbn="", publisher="", entry_type=:article),
    # Variable selection — stepwise / GETS (EV-04, #412)
    :hoover_perez1999 => (key=:hoover_perez1999,
        authors="Hoover, Kevin D. and Perez, Stephen J.", year=1999,
        title="Data Mining Reconsidered: Encompassing and the General-to-Specific Approach to Specification Search",
        journal="Econometrics Journal", volume="2", issue="2", pages="167--191",
        doi="10.1111/1368-423X.00025", isbn="", publisher="", entry_type=:article),
    :hendry_krolzig2005 => (key=:hendry_krolzig2005,
        authors="Hendry, David F. and Krolzig, Hans-Martin", year=2005,
        title="The Properties of Automatic Gets Modelling",
        journal="Economic Journal", volume="115", issue="502", pages="C32--C61",
        doi="10.1111/j.0013-0133.2005.00979.x", isbn="", publisher="", entry_type=:article),
    :pretis2018 => (key=:pretis2018,
        authors="Pretis, Felix and Reade, J. James and Sucarrat, Genaro", year=2018,
        title="Automated General-to-Specific (GETS) Regression Modeling and Indicator Saturation for Outliers and Structural Breaks",
        journal="Journal of Statistical Software", volume="86", issue="3", pages="1--44",
        doi="10.18637/jss.v086.i03", isbn="", publisher="", entry_type=:article),
    :friedman2010 => (key=:friedman2010,
        authors="Friedman, Jerome and Hastie, Trevor and Tibshirani, Robert", year=2010,
        title="Regularization Paths for Generalized Linear Models via Coordinate Descent",
        journal="Journal of Statistical Software", volume="33", issue="1", pages="1--22",
        doi="10.18637/jss.v033.i01", isbn="", publisher="", entry_type=:article),
    :belloni_chernozhukov2013 => (key=:belloni_chernozhukov2013,
        authors="Belloni, Alexandre and Chernozhukov, Victor", year=2013,
        title="Least Squares After Model Selection in High-Dimensional Sparse Models",
        journal="Bernoulli", volume="19", issue="2", pages="521--547",
        doi="10.3150/11-BEJ410", isbn="", publisher="", entry_type=:article),
    # --- Censored / Truncated Regression (EV-17, #425) ---
    :tobin1958 => (key=:tobin1958, authors="Tobin, James", year=1958,
        title="Estimation of Relationships for Limited Dependent Variables",
        journal="Econometrica", volume="26", issue="1", pages="24--36",
        doi="10.2307/1907382", isbn="", publisher="", entry_type=:article),
    :olsen1978 => (key=:olsen1978, authors="Olsen, Randall J.", year=1978,
        title="Note on the Uniqueness of the Maximum Likelihood Estimator for the Tobit Model",
        journal="Econometrica", volume="46", issue="5", pages="1211--1215",
        doi="10.2307/1911445", isbn="", publisher="", entry_type=:article),
    :hausman_wise1977 => (key=:hausman_wise1977, authors="Hausman, Jerry A. and Wise, David A.", year=1977,
        title="Social Experimentation, Truncated Distributions, and Efficient Estimation",
        journal="Econometrica", volume="45", issue="4", pages="919--938",
        doi="10.2307/1912682", isbn="", publisher="", entry_type=:article),
    :mcdonald_moffitt1980 => (key=:mcdonald_moffitt1980,
        authors="McDonald, John F. and Moffitt, Robert A.", year=1980,
        title="The Uses of Tobit Analysis",
        journal="The Review of Economics and Statistics", volume="62", issue="2", pages="318--321",
        doi="10.2307/1924766", isbn="", publisher="", entry_type=:article),
    # --- Sample-selection / Heckman model (EV-18, #426) ---
    :heckman1979 => (key=:heckman1979, authors="Heckman, James J.", year=1979,
        title="Sample Selection Bias as a Specification Error",
        journal="Econometrica", volume="47", issue="1", pages="153--161",
        doi="10.2307/1912352", isbn="", publisher="", entry_type=:article),
    :mroz1987 => (key=:mroz1987, authors="Mroz, Thomas A.", year=1987,
        title="The Sensitivity of an Empirical Model of Married Women's Hours of Work to Economic and Statistical Assumptions",
        journal="Econometrica", volume="55", issue="4", pages="765--799",
        doi="10.2307/1911029", isbn="", publisher="", entry_type=:article),
    :greene2018 => (key=:greene2018, authors="Greene, William H.", year=2018,
        title="Econometric Analysis", journal="", volume="", issue="", pages="",
        doi="", isbn="978-0134461366", publisher="Pearson (8th ed.)", entry_type=:book),
    # --- Robust regression: M / MM estimation (EV-40, #448) ---
    :huber1964 => (key=:huber1964, authors="Huber, Peter J.", year=1964,
        title="Robust Estimation of a Location Parameter",
        journal="The Annals of Mathematical Statistics", volume="35", issue="1", pages="73--101",
        doi="10.1214/aoms/1177703732", isbn="", publisher="", entry_type=:article),
    :yohai1987 => (key=:yohai1987, authors="Yohai, Víctor J.", year=1987,
        title="High Breakdown-Point and High Efficiency Robust Estimates for Regression",
        journal="The Annals of Statistics", volume="15", issue="2", pages="642--656",
        doi="10.1214/aos/1176350366", isbn="", publisher="", entry_type=:article),
    :salibian_yohai2006 => (key=:salibian_yohai2006,
        authors="Salibian-Barrera, Matías and Yohai, Víctor J.", year=2006,
        title="A Fast Algorithm for S-Regression Estimates",
        journal="Journal of Computational and Graphical Statistics", volume="15", issue="2", pages="414--427",
        doi="10.1198/106186006X113629", isbn="", publisher="", entry_type=:article),
    :huber_ronchetti2009 => (key=:huber_ronchetti2009,
        authors="Huber, Peter J. and Ronchetti, Elvezio M.", year=2009,
        title="Robust Statistics", journal="", volume="", issue="", pages="",
        doi="10.1002/9780470434697", isbn="978-0-470-12990-6", publisher="Wiley (2nd ed.)", entry_type=:book),
    :brownlee1965 => (key=:brownlee1965, authors="Brownlee, Kenneth A.", year=1965,
        title="Statistical Theory and Methodology in Science and Engineering",
        journal="", volume="", issue="", pages="491--500",
        doi="", isbn="", publisher="Wiley (2nd ed.)", entry_type=:book),
    # --- Single-Equation Cointegrating Regression (EV-10, #418) ---
    :phillips_hansen1990 => (key=:phillips_hansen1990,
        authors="Phillips, Peter C. B. and Hansen, Bruce E.", year=1990,
        title="Statistical Inference in Instrumental Variables Regression with I(1) Processes",
        journal="The Review of Economic Studies", volume="57", issue="1", pages="99--125",
        doi="10.2307/2297545", isbn="", publisher="", entry_type=:article),
    :park1992 => (key=:park1992, authors="Park, Joon Y.", year=1992,
        title="Canonical Cointegrating Regressions",
        journal="Econometrica", volume="60", issue="1", pages="119--143",
        doi="10.2307/2951679", isbn="", publisher="", entry_type=:article),
    # --- Residual-based / parameter-stability cointegration tests (EV-11) ---
    :phillips_ouliaris1990 => (key=:phillips_ouliaris1990,
        authors="Phillips, Peter C. B. and Ouliaris, Sam", year=1990,
        title="Asymptotic Properties of Residual Based Tests for Cointegration",
        journal="Econometrica", volume="58", issue="1", pages="165--193",
        doi="10.2307/2938339", isbn="", publisher="", entry_type=:article),
    :hansen1992_instability => (key=:hansen1992_instability,
        authors="Hansen, Bruce E.", year=1992,
        title="Tests for Parameter Instability in Regressions with I(1) Processes",
        journal="Journal of Business & Economic Statistics", volume="10", issue="3",
        pages="321--335", doi="10.1080/07350015.1992.10509908", isbn="", publisher="",
        entry_type=:article),
    :park1990_added => (key=:park1990_added, authors="Park, Joon Y.", year=1990,
        title="Testing for Unit Roots and Cointegration by Adding Superfluous Regressors",
        journal="Center for Analytic Economics Working Paper, Cornell University",
        volume="", issue="", pages="", doi="", isbn="", publisher="", entry_type=:techreport),
    :mackinnon2010 => (key=:mackinnon2010, authors="MacKinnon, James G.", year=2010,
        title="Critical Values for Cointegration Tests",
        journal="Queen's University Department of Economics Working Paper", volume="1227",
        issue="", pages="", doi="", isbn="", publisher="", entry_type=:techreport),
    :saikkonen1991 => (key=:saikkonen1991, authors="Saikkonen, Pentti", year=1991,
        title="Asymptotically Efficient Estimation of Cointegration Regressions",
        journal="Econometric Theory", volume="7", issue="1", pages="1--21",
        doi="10.1017/S0266466600004217", isbn="", publisher="", entry_type=:article),
    :stock_watson1993 => (key=:stock_watson1993,
        authors="Stock, James H. and Watson, Mark W.", year=1993,
        title="A Simple Estimator of Cointegrating Vectors in Higher Order Integrated Systems",
        journal="Econometrica", volume="61", issue="4", pages="783--820",
        doi="10.2307/2951763", isbn="", publisher="", entry_type=:article),
    # --- Ordered & Multinomial Models ---
    :mccullagh1980 => (key=:mccullagh1980, authors="McCullagh, Peter", year=1980,
        title="Regression Models for Ordinal Data",
        journal="Journal of the Royal Statistical Society: Series B", volume="42", issue="2", pages="109--142",
        doi="10.1111/j.2517-6161.1980.tb01109.x", isbn="", publisher="", entry_type=:article),
    :brant1990 => (key=:brant1990, authors="Brant, Rollin", year=1990,
        title="Assessing Proportionality in the Proportional Odds Model",
        journal="Biometrics", volume="46", issue="4", pages="1171--1178",
        doi="10.2307/2532457", isbn="", publisher="", entry_type=:article),
    :hausman_mcfadden1984 => (key=:hausman_mcfadden1984, authors="Hausman, Jerry and McFadden, Daniel", year=1984,
        title="Specification Tests for the Multinomial Logit Model",
        journal="Econometrica", volume="52", issue="5", pages="1219--1240",
        doi="10.2307/1910997", isbn="", publisher="", entry_type=:article),
    # --- Panel Regression ---
    :baltagi2021 => (key=:baltagi2021, authors="Baltagi, Badi H.", year=2021,
        title="Econometric Analysis of Panel Data", journal="",
        volume="", issue="", pages="", doi="",
        isbn="978-3-030-53952-8", publisher="Springer", entry_type=:book),
    :hausman_taylor1981 => (key=:hausman_taylor1981,
        authors="Hausman, Jerry A. and Taylor, William E.", year=1981,
        title="Panel Data and Unobservable Individual Effects",
        journal="Econometrica", volume="49", issue="6", pages="1377--1398",
        doi="10.2307/1911406", isbn="", publisher="", entry_type=:article),
    :chamberlain1980 => (key=:chamberlain1980, authors="Chamberlain, Gary", year=1980,
        title="Analysis of Covariance with Qualitative Data",
        journal="Review of Economic Studies", volume="47", issue="1", pages="225--238",
        doi="10.2307/2297110", isbn="", publisher="", entry_type=:article),
    :mundlak1978 => (key=:mundlak1978, authors="Mundlak, Yair", year=1978,
        title="On the Pooling of Time Series and Cross Section Data",
        journal="Econometrica", volume="46", issue="1", pages="69--85",
        doi="10.2307/1913646", isbn="", publisher="", entry_type=:article),
    :breusch_pagan1980 => (key=:breusch_pagan1980, authors="Breusch, Trevor S. and Pagan, Adrian R.", year=1980,
        title="The Lagrange Multiplier Test and Its Applications to Model Specification in Econometrics",
        journal="Review of Economic Studies", volume="47", issue="1", pages="239--253",
        doi="10.2307/2297111", isbn="", publisher="", entry_type=:article),
    :driscoll_kraay1998 => (key=:driscoll_kraay1998,
        authors="Driscoll, John C. and Kraay, Aart C.", year=1998,
        title="Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data",
        journal="Review of Economics and Statistics", volume="80", issue="4", pages="549--560",
        doi="10.1162/003465398557825", isbn="", publisher="", entry_type=:article),
    # --- MIDAS Regression (EV-01) ---
    :ghysels2007 => (key=:ghysels2007,
        authors="Ghysels, Eric and Sinko, Arthur and Valkanov, Rossen", year=2007,
        title="MIDAS Regressions: Further Results and New Directions",
        journal="Econometric Reviews", volume="26", issue="1", pages="53--90",
        doi="10.1080/07474930600972467", isbn="", publisher="", entry_type=:article),
    :andreou2010 => (key=:andreou2010,
        authors="Andreou, Elena and Ghysels, Eric and Kourtellos, Andros", year=2010,
        title="Regression Models with Mixed Sampling Frequencies",
        journal="Journal of Econometrics", volume="158", issue="2", pages="246--261",
        doi="10.1016/j.jeconom.2010.01.004", isbn="", publisher="", entry_type=:article),
    :foroni2015 => (key=:foroni2015,
        authors="Foroni, Claudia and Marcellino, Massimiliano and Schumacher, Christian", year=2015,
        title="Unrestricted Mixed Data Sampling (MIDAS): MIDAS Regressions with Unrestricted Lag Polynomials",
        journal="Journal of the Royal Statistical Society: Series A", volume="178", issue="1", pages="57--82",
        doi="10.1111/rssa.12043", isbn="", publisher="", entry_type=:article),
    # --- ARDL & Bounds Test (EV-08, #416) ---
    :pesaran_shin1999 => (key=:pesaran_shin1999,
        authors="Pesaran, M. Hashem and Shin, Yongcheol", year=1999,
        title="An Autoregressive Distributed Lag Modelling Approach to Cointegration Analysis",
        journal="", volume="", issue="", pages="371--413", doi="",
        isbn="978-0-521-63323-9",
        publisher="Cambridge University Press (Strom, ed., Econometrics and Economic Theory in the 20th Century)",
        entry_type=:incollection),
    :pesaran_shin_smith2001 => (key=:pesaran_shin_smith2001,
        authors="Pesaran, M. Hashem and Shin, Yongcheol and Smith, Richard J.", year=2001,
        title="Bounds Testing Approaches to the Analysis of Level Relationships",
        journal="Journal of Applied Econometrics", volume="16", issue="3", pages="289--326",
        doi="10.1002/jae.616", isbn="", publisher="", entry_type=:article),
    :narayan2005 => (key=:narayan2005, authors="Narayan, Paresh Kumar", year=2005,
        title="The Saving and Investment Nexus for China: Evidence from Cointegration Tests",
        journal="Applied Economics", volume="37", issue="17", pages="1979--1990",
        doi="10.1080/00036840500278103", isbn="", publisher="", entry_type=:article),
    :kripfganz_schneider2023 => (key=:kripfganz_schneider2023,
        authors="Kripfganz, Sebastian and Schneider, Daniel C.", year=2023,
        title="ARDL: Estimating Autoregressive Distributed Lag and Equilibrium Correction Models",
        journal="Stata Journal", volume="23", issue="4", pages="983--1019",
        doi="10.1177/1536867X231212434", isbn="", publisher="", entry_type=:article),
    # --- Nonlinear ARDL (NARDL, EV-09, #417) ---
    :shin_yu_greenwood_nimmo2014 => (key=:shin_yu_greenwood_nimmo2014,
        authors="Shin, Yongcheol and Yu, Byungchul and Greenwood-Nimmo, Matthew", year=2014,
        title="Modelling Asymmetric Cointegration and Dynamic Multipliers in a Nonlinear ARDL Framework",
        journal="", volume="", issue="", pages="281--314", doi="10.1007/978-1-4899-8008-3_9",
        isbn="978-1-4899-8007-6",
        publisher="Springer (Sickles and Horrace, eds., Festschrift in Honor of Peter Schmidt)",
        entry_type=:incollection),
    # --- Panel ARDL: PMG / MG / DFE (EV-23, #431) ---
    :pesaran_shin_smith1999 => (key=:pesaran_shin_smith1999,
        authors="Pesaran, M. Hashem and Shin, Yongcheol and Smith, Ron P.", year=1999,
        title="Pooled Mean Group Estimation of Dynamic Heterogeneous Panels",
        journal="Journal of the American Statistical Association", volume="94", issue="446",
        pages="621--634", doi="10.1080/01621459.1999.10474156", isbn="", publisher="",
        entry_type=:article),
    :pesaran_smith1995 => (key=:pesaran_smith1995,
        authors="Pesaran, M. Hashem and Smith, Ron", year=1995,
        title="Estimating Long-Run Relationships from Dynamic Heterogeneous Panels",
        journal="Journal of Econometrics", volume="68", issue="1", pages="79--113",
        doi="10.1016/0304-4076(94)01644-F", isbn="", publisher="", entry_type=:article),
    :blackburne_frank2007 => (key=:blackburne_frank2007,
        authors="Blackburne, Edward F. and Frank, Mark W.", year=2007,
        title="Estimation of Nonstationary Heterogeneous Panels",
        journal="Stata Journal", volume="7", issue="2", pages="197--208",
        doi="10.1177/1536867X0700700204", isbn="", publisher="", entry_type=:article),
    # --- Forecast evaluation & combination (EV-39, #447) ---
    :diebold_mariano1995 => (key=:diebold_mariano1995,
        authors="Diebold, Francis X. and Mariano, Roberto S.", year=1995,
        title="Comparing Predictive Accuracy",
        journal="Journal of Business \\& Economic Statistics", volume="13", issue="3",
        pages="253--263", doi="10.1080/07350015.1995.10524599",
        isbn="", publisher="", entry_type=:article),
    :harvey_leybourne_newbold1997 => (key=:harvey_leybourne_newbold1997,
        authors="Harvey, David and Leybourne, Stephen and Newbold, Paul", year=1997,
        title="Testing the Equality of Prediction Mean Squared Errors",
        journal="International Journal of Forecasting", volume="13", issue="2",
        pages="281--291", doi="10.1016/S0169-2070(96)00719-4",
        isbn="", publisher="", entry_type=:article),
    :harvey_leybourne_newbold1998 => (key=:harvey_leybourne_newbold1998,
        authors="Harvey, David I. and Leybourne, Stephen J. and Newbold, Paul", year=1998,
        title="Tests for Forecast Encompassing",
        journal="Journal of Business \\& Economic Statistics", volume="16", issue="2",
        pages="254--259", doi="10.1080/07350015.1998.10524759",
        isbn="", publisher="", entry_type=:article),
    :clark_west2007 => (key=:clark_west2007,
        authors="Clark, Todd E. and West, Kenneth D.", year=2007,
        title="Approximately Normal Tests for Equal Predictive Accuracy in Nested Models",
        journal="Journal of Econometrics", volume="138", issue="1", pages="291--311",
        doi="10.1016/j.jeconom.2006.05.023", isbn="", publisher="", entry_type=:article),
    :bates_granger1969 => (key=:bates_granger1969,
        authors="Bates, J. M. and Granger, C. W. J.", year=1969,
        title="The Combination of Forecasts",
        journal="Operational Research Quarterly", volume="20", issue="4", pages="451--468",
        doi="10.1057/jors.1969.103", isbn="", publisher="", entry_type=:article),
    :granger_ramanathan1984 => (key=:granger_ramanathan1984,
        authors="Granger, C. W. J. and Ramanathan, Ramu", year=1984,
        title="Improved Methods of Combining Forecasts",
        journal="Journal of Forecasting", volume="3", issue="2", pages="197--204",
        doi="10.1002/for.3980030207", isbn="", publisher="", entry_type=:article),
    :mincer_zarnowitz1969 => (key=:mincer_zarnowitz1969,
        authors="Mincer, Jacob A. and Zarnowitz, Victor", year=1969,
        title="The Evaluation of Economic Forecasts",
        journal="", volume="", issue="", pages="3--46", doi="",
        isbn="0-870-14202-X",
        publisher="NBER (Mincer, ed., Economic Forecasts and Expectations)",
        entry_type=:incollection),
    :theil1966 => (key=:theil1966,
        authors="Theil, Henri", year=1966,
        title="Applied Economic Forecasting",
        journal="", volume="", issue="", pages="", doi="", isbn="",
        publisher="North-Holland, Amsterdam", entry_type=:book),
    :hyndman_koehler2006 => (key=:hyndman_koehler2006,
        authors="Hyndman, Rob J. and Koehler, Anne B.", year=2006,
        title="Another Look at Measures of Forecast Accuracy",
        journal="International Journal of Forecasting", volume="22", issue="4",
        pages="679--688", doi="10.1016/j.ijforecast.2006.03.001",
        isbn="", publisher="", entry_type=:article),
    # --- Dumitrescu-Hurlin panel Granger non-causality (EV-24, #432) ---
    :dumitrescu_hurlin2012 => (key=:dumitrescu_hurlin2012,
        authors="Dumitrescu, Elena-Ivona and Hurlin, Christophe", year=2012,
        title="Testing for Granger Non-causality in Heterogeneous Panels",
        journal="Economic Modelling", volume="29", issue="4", pages="1450--1460",
        doi="10.1016/j.econmod.2012.02.014", isbn="", publisher="", entry_type=:article),
    # --- EDF goodness-of-fit battery (EV-26, #434) ---
    :anderson_darling1954 => (key=:anderson_darling1954,
        authors="Anderson, Theodore W. and Darling, Donald A.", year=1954,
        title="A Test of Goodness of Fit",
        journal="Journal of the American Statistical Association", volume="49", issue="268",
        pages="765--769", doi="10.1080/01621459.1954.10501232", isbn="", publisher="",
        entry_type=:article),
    :stephens1974 => (key=:stephens1974, authors="Stephens, Michael A.", year=1974,
        title="EDF Statistics for Goodness of Fit and Some Comparisons",
        journal="Journal of the American Statistical Association", volume="69", issue="347",
        pages="730--737", doi="10.1080/01621459.1974.10480196", isbn="", publisher="",
        entry_type=:article),
    :lilliefors1967 => (key=:lilliefors1967, authors="Lilliefors, Hubert W.", year=1967,
        title="On the Kolmogorov-Smirnov Test for Normality with Mean and Variance Unknown",
        journal="Journal of the American Statistical Association", volume="62", issue="318",
        pages="399--402", doi="10.1080/01621459.1967.10482916", isbn="", publisher="",
        entry_type=:article),
    :dallal_wilkinson1986 => (key=:dallal_wilkinson1986,
        authors="Dallal, Gerard E. and Wilkinson, Leland", year=1986,
        title="An Analytic Approximation to the Distribution of Lilliefors's Test Statistic for Normality",
        journal="The American Statistician", volume="40", issue="4", pages="294--296",
        doi="10.1080/00031305.1986.10475419", isbn="", publisher="", entry_type=:article),
    :marsaglia_tsang_wang2003 => (key=:marsaglia_tsang_wang2003,
        authors="Marsaglia, George and Tsang, Wai Wan and Wang, Jingbo", year=2003,
        title="Evaluating Kolmogorov's Distribution",
        journal="Journal of Statistical Software", volume="8", issue="18", pages="1--4",
        doi="10.18637/jss.v008.i18", isbn="", publisher="", entry_type=:article),
)

# --- Type/method → reference keys mapping ---

const _TYPE_REFS = Dict{Symbol, Vector{Symbol}}(
    # Long-run variance toolkit (EV-12): lrvar/lrcov/lrcov_oneside/varhac
    :lrvar => [:andrews1991, :newey_west1994, :andrews_monahan1992, :den_haan_levin1997],
    # Input-Output analysis
    :IOData => [:leontief1936, :ghosh1958, :miller_blair_2009],
    :LeontiefModel => [:leontief1936, :miller_blair_2009],
    :GhoshModel => [:ghosh1958, :miller_blair_2009],
    :IOMultipliers => [:miller_blair_2009],
    :LinkageResult => [:rasmussen1956, :hirschman1958, :miller_blair_2009],
    :SDAResult => [:dietzenbacher_los1998, :miller_blair_2009],
    :ExtractionResult => [:miller_blair_2009],
    :FootprintResult => [:leontief1936, :miller_blair_2009],
    :BaqaeeFarhiResult => [:baqaee_farhi_2019, :miller_blair_2009],
    :io => [:leontief1936, :ghosh1958, :miller_blair_2009, :baqaee_farhi_2019],
    # VAR
    :VARModel => [:sims1980, :lutkepohl2005],
    :ImpulseResponse => [:lutkepohl2005, :kilian1998],
    :BayesianImpulseResponse => [:lutkepohl2005, :kilian1998],
    :FEVD => [:lutkepohl2005],
    :BayesianFEVD => [:lutkepohl2005],
    :HistoricalDecomposition => [:kilian_lutkepohl2017],
    :BayesianHistoricalDecomposition => [:kilian_lutkepohl2017],
    :AriasSVARResult => [:arias_rubio_ramirez_waggoner2018],
    :UhligSVARResult => [:mountford_uhlig2009, :uhlig2005],
    :SVARRestrictions => [:arias_rubio_ramirez_waggoner2018],
    :SignIdentifiedSet => [:rubio_ramirez2010, :baumeister_hamilton2015],
    # Bayesian VAR
    :MinnesotaHyperparameters => [:litterman1986, :kadiyala_karlsson1997],
    :BVARPosterior => [:litterman1986, :kadiyala_karlsson1997, :giannone_lenza_primiceri2015],
    :bvar => [:litterman1986, :kadiyala_karlsson1997, :giannone_lenza_primiceri2015],
    # Identification methods (symbol dispatch)
    :cholesky => [:sims1980, :lutkepohl2005],
    :long_run => [:blanchard_quah1989],
    :sign => [:uhlig2005],
    :narrative => [:antolin_diaz_rubio_ramirez2018],
    :arias => [:arias_rubio_ramirez_waggoner2018],
    # Local Projections
    :LPModel => [:jorda2005],
    :LPImpulseResponse => [:jorda2005],
    :LPIVModel => [:stock_watson2018],
    :SmoothLPModel => [:barnichon_brownlees2019],
    :StateLPModel => [:auerbach_gorodnichenko2012],
    :PropensityLPModel => [:angrist_jorda_kuersteiner2018],
    :StructuralLP => [:plagborg_moller_wolf2021, :jorda2005],
    :LPForecast => [:jorda2005],
    :LPFEVD => [:gorodnichenko_lee2020],
    # Factor Models
    :FactorModel => [:bai_ng2002, :stock_watson2002],
    :DynamicFactorModel => [:stock_watson2002],
    :GeneralizedDynamicFactorModel => [:stock_watson2002],
    :FactorForecast => [:stock_watson2002],
    # FAVAR & Structural DFM
    :FAVARModel => [:bernanke_boivin_eliasz2005, :stock_watson2002],
    :BayesianFAVAR => [:bernanke_boivin_eliasz2005, :stock_watson2002],
    :StructuralDFM => [:forni_giannone_lippi_reichlin2009, :stock_watson2002],
    :favar => [:bernanke_boivin_eliasz2005, :stock_watson2002],
    :structural_dfm => [:forni_giannone_lippi_reichlin2009, :stock_watson2002],
    :sdfm_panel_irf => [:forni_giannone_lippi_reichlin2009, :stock_watson2002],
    # Unit Root Tests
    :ADFResult => [:dickey_fuller1979],
    :KPSSResult => [:kpss1992],
    :PPResult => [:phillips_perron1988],
    :ZAResult => [:zivot_andrews1992],
    :NgPerronResult => [:ng_perron2001],
    :JohansenResult => [:johansen1991],
    :adf => [:dickey_fuller1979],
    :kpss => [:kpss1992],
    :pp => [:phillips_perron1988],
    :za => [:zivot_andrews1992],
    :ngperron => [:ng_perron2001],
    :johansen => [:johansen1991],
    # VECM
    :VECMModel => [:johansen1991, :engle_granger1987, :lutkepohl2005],
    :VECMForecast => [:johansen1991, :lutkepohl2005],
    :VECMGrangerResult => [:johansen1991, :lutkepohl2005],
    :vecm => [:johansen1991, :engle_granger1987, :lutkepohl2005],
    :engle_granger => [:engle_granger1987],
    # ARIMA
    :ARModel => [:box_jenkins1970],
    :MAModel => [:box_jenkins1970],
    :ARMAModel => [:box_jenkins1970],
    :ARIMAModel => [:box_jenkins1970],
    :ARIMAForecast => [:box_jenkins1970],
    :ARIMAOrderSelection => [:hyndman_khandakar2008],
    :auto_arima => [:hyndman_khandakar2008],
    # ARFIMA / long memory (EV-13)
    :ARFIMAModel => [:sowell1992, :hosking1981, :jensen_nielsen2014],
    :estimate_arfima => [:sowell1992, :hosking1981, :jensen_nielsen2014],
    :GPHResult => [:geweke_porter_hudak1983],
    :gph_test => [:geweke_porter_hudak1983],
    :LocalWhittleResult => [:robinson1995],
    :local_whittle => [:robinson1995],
    # MIDAS (EV-01)
    :MidasModel => [:ghysels2007, :andreou2010, :foroni2015],
    :MidasForecast => [:ghysels2007, :andreou2010],
    :midas => [:ghysels2007, :andreou2010, :foroni2015],
    # ARDL & bounds test (EV-08, #416)
    :ARDLModel => [:pesaran_shin1999, :pesaran_shin_smith2001, :kripfganz_schneider2023],
    :ARDLBoundsTest => [:pesaran_shin_smith2001, :narayan2005],
    :ardl => [:pesaran_shin1999, :pesaran_shin_smith2001, :narayan2005, :kripfganz_schneider2023],
    # NARDL — nonlinear ARDL (EV-09, #417)
    :NARDLModel => [:shin_yu_greenwood_nimmo2014, :pesaran_shin_smith2001],
    :NARDLSymmetryTest => [:shin_yu_greenwood_nimmo2014],
    :NARDLMultipliers => [:shin_yu_greenwood_nimmo2014],
    :nardl => [:shin_yu_greenwood_nimmo2014, :pesaran_shin_smith2001],
    # Panel ARDL: PMG / MG / DFE (EV-23, #431)
    :PMGModel => [:pesaran_shin_smith1999, :pesaran_smith1995, :blackburne_frank2007],
    :estimate_pmg => [:pesaran_shin_smith1999, :pesaran_smith1995, :blackburne_frank2007],
    # Forecast evaluation & combination (EV-39, #447)
    :ForecastEvaluation => [:theil1966, :hyndman_koehler2006],
    :DMTestResult => [:diebold_mariano1995, :harvey_leybourne_newbold1997],
    :ClarkWestResult => [:clark_west2007],
    :MincerZarnowitzResult => [:mincer_zarnowitz1969],
    :ForecastEncompassingResult => [:harvey_leybourne_newbold1998],
    :ForecastCombination => [:bates_granger1969, :granger_ramanathan1984],
    # GMM
    :GMMModel => [:hansen1982],
    :gmm => [:hansen1982],
    # SMM
    :SMMModel => [:ruge_murcia2012, :lee_ingram1991, :hansen1982],
    :ParameterTransform => [:hansen1982],
    :smm => [:ruge_murcia2012, :lee_ingram1991, :hansen1982],
    # Analytical GMM
    :analytical_gmm => [:hamilton1994, :hansen1982],
    # Non-Gaussian ICA methods (symbol dispatch)
    :fastica => [:hyvarinen1999, :lewis2025],
    :jade => [:cardoso_souloumiac1993, :lewis2025],
    :sobi => [:belouchrani1997, :lewis2025],
    :dcov => [:szekely_rizzo_bakirov2007, :matteson_tsay2017, :lewis2025],
    :hsic => [:gretton2005, :lewis2025],
    # Non-Gaussian ML methods (symbol dispatch)
    :student_t => [:lanne_meitz_saikkonen2017, :lewis2025],
    :mixture_normal => [:lanne_meitz_saikkonen2017, :lewis2025],
    :pml => [:lanne_meitz_saikkonen2017, :lewis2025],
    :skew_normal => [:lanne_meitz_saikkonen2017, :lewis2025],
    :nongaussian_ml => [:lanne_meitz_saikkonen2017, :lewis2025],
    # Non-Gaussian result types
    :ICASVARResult => [:lanne_meitz_saikkonen2017, :lewis2025],
    :NonGaussianMLResult => [:lanne_meitz_saikkonen2017, :lewis2025],
    # Heteroskedastic identification
    :MarkovSwitchingSVARResult => [:rigobon2003, :lanne_lutkepohl2008, :lewis2025],
    :GARCHSVARResult => [:rigobon2003, :normandin_phaneuf2004, :lewis2025],
    :SmoothTransitionSVARResult => [:lutkepohl_netsunajev2017, :lewis2025],
    :ExternalVolatilitySVARResult => [:rigobon2003, :lewis2025],
    :markov_switching => [:rigobon2003, :lanne_lutkepohl2008, :lewis2025],
    :smooth_transition => [:lutkepohl_netsunajev2017, :lewis2025],
    :external_volatility => [:rigobon2003, :lewis2025],
    # Normality tests
    :NormalityTestResult => [:jarque_bera1980, :mardia1970],
    :NormalityTestSuite => [:jarque_bera1980, :mardia1970, :doornik_hansen2008, :henze_zirkler1990],
    :jarque_bera => [:jarque_bera1980],
    :mardia => [:mardia1970],
    :doornik_hansen => [:doornik_hansen2008],
    :henze_zirkler => [:henze_zirkler1990],
    # Covariance estimators
    :NeweyWestEstimator => [:newey_west1987],
    :WhiteEstimator => [:white1980],
    :newey_west => [:newey_west1987],
    :white => [:white1980],
    # Volatility models
    :ARCHModel => [:engle1982],
    :GARCHModel => [:bollerslev1986],
    :EGARCHModel => [:nelson1991],
    :GJRGARCHModel => [:glosten_jagannathan_runkle1993],
    :GarchMidasModel => [:engle_ghysels_sohn2013, :ghysels2007, :bollerslev1986],
    :garch_midas => [:engle_ghysels_sohn2013, :ghysels2007, :bollerslev1986],
    :FIGARCHModel => [:baillie_bollerslev_mikkelsen1996, :bollerslev1986],
    :FIEGARCHModel => [:bollerslev_mikkelsen1996, :nelson1991],
    :IGARCHModel => [:engle_bollerslev1986, :bollerslev1986],
    :CGARCHModel => [:engle_lee1999, :bollerslev1986],
    :APARCHModel => [:ding_granger_engle1993, :glosten_jagannathan_runkle1993],
    :MGARCHModel => [:bollerslev1990, :engle2002dcc, :aielli2013, :engle_kroner1995],
    :SVModel => [:taylor1986, :kim_shephard_chib1998, :omori2007],
    :VolatilityForecast => [:engle1982, :bollerslev1986],
    :arch => [:engle1982],
    :garch => [:bollerslev1986],
    :egarch => [:nelson1991],
    :gjr_garch => [:glosten_jagannathan_runkle1993],
    :sv => [:taylor1986, :kim_shephard_chib1998, :omori2007],
    # Nonlinear time series (threshold/SETAR)
    :ThresholdModel => [:tong1990, :hansen1996, :hansen2000],
    :HansenLinearityTest => [:hansen1996],
    :threshold => [:tong1990, :hansen2000],
    :setar => [:tong1990, :hansen2000],
    # Smooth-transition autoregression (STAR) — EV-06
    :STARModel => [:luukkonen1988, :terasvirta1994],
    :star => [:luukkonen1988, :terasvirta1994],
    # Markov-switching regression / MS-AR — EV-07
    :MSRegModel => [:hamilton1989, :kim1994, :hamilton1994],
    :ms => [:hamilton1989, :kim1994],
    :ms_ar => [:hamilton1989, :kim1994],
    # Time Series Filters
    :HPFilterResult => [:hodrick_prescott1997],
    :HamiltonFilterResult => [:hamilton2018filter],
    :BeveridgeNelsonResult => [:beveridge_nelson1981, :morley_nelson_zivot2003],
    :BaxterKingResult => [:baxter_king1999],
    :BoostedHPResult => [:phillips_shi2021, :mei_phillips_shi2024],
    :hp_filter => [:hodrick_prescott1997],
    :hamilton_filter => [:hamilton2018filter],
    :beveridge_nelson => [:beveridge_nelson1981, :morley_nelson_zivot2003],
    :baxter_king => [:baxter_king1999],
    :boosted_hp => [:phillips_shi2021, :mei_phillips_shi2024],
    :X13FilterResult => [:dagum_bianconcini2016, :findley1998, :gomez_maravall1996],
    :x13_filter => [:dagum_bianconcini2016, :findley1998, :gomez_maravall1996],
    # Model comparison tests
    :LRTestResult => [:wilks1938, :neyman_pearson1933],
    :LMTestResult => [:rao1948, :silvey1959],
    :lr_test => [:wilks1938, :neyman_pearson1933],
    :lm_test => [:rao1948, :silvey1959],
    # Granger causality
    :GrangerCausalityResult => [:granger1969, :lutkepohl2005],
    :granger => [:granger1969, :lutkepohl2005],
    :granger_test => [:granger1969, :lutkepohl2005],
    # Panel VAR
    :PVARModel => [:holtz_eakin1988, :arellano_bond1991, :blundell_bond1998],
    :PVARStability => [:holtz_eakin1988],
    :PVARTestResult => [:hansen1982],
    :pvar => [:holtz_eakin1988, :arellano_bond1991, :blundell_bond1998],
    :fd_gmm => [:arellano_bond1991],
    :system_gmm => [:blundell_bond1998],
    :windmeijer => [:windmeijer2005],
    :andrews_lu => [:andrews_lu2001],
    :girf => [:pesaran_shin1998],
    # Nowcasting
    :NowcastDFM => [:banbura_modugno2014, :delle_chiaie2022],
    :NowcastBVAR => [:cimadomo2022],
    :NowcastBridge => [:banbura2023],
    :NowcastResult => [:banbura_modugno2014],
    :NowcastNews => [:banbura_modugno2014],
    :nowcast_dfm => [:banbura_modugno2014, :delle_chiaie2022],
    :nowcast_bvar => [:cimadomo2022],
    :nowcast_bridge => [:banbura2023],
    :nowcast_news => [:banbura_modugno2014],
    :balance_panel => [:banbura_modugno2014],
    # DSGE
    :DSGESolution => [:sims2002, :blanchard_kahn1980],
    :DSGEEstimation => [:sims2002, :christiano_eichenbaum_evans2005, :hansen_singleton1982, :smets_wouters2007],
    :PerfectForesightPath => [:sims2002],
    :DSGESpec => [:sims2002, :fernandez_villaverde_rubio_schorfheide2016],
    :gensys => [:sims2002],
    :blanchard_kahn => [:blanchard_kahn1980],
    :perfect_foresight => [:sims2002],
    :irf_matching => [:christiano_eichenbaum_evans2005],
    :euler_gmm => [:hansen_singleton1982, :hansen1982],
    # OccBin
    :OccBinSolution => [:guerrieri_iacoviello2015],
    :OccBinIRF => [:guerrieri_iacoviello2015],
    :occbin => [:guerrieri_iacoviello2015],
    :occbin_solve => [:guerrieri_iacoviello2015],
    :occbin_irf => [:guerrieri_iacoviello2015],
    # DSGE solver methods
    :klein => [:klein2000],
    :perturbation_solver => [:schmitt_grohe_uribe2004, :kim_kim_schaumburg_sims2008],
    :collocation_solver => [:judd1998, :judd_maliar_maliar_valero2014],
    :pfi_solver => [:coleman1990, :judd1998],
    :vfi_solver => [:stokey_lucas_prescott1989, :howard1960, :judd1998, :santos_rust2003],
    :_anderson_step => [:walker_ni2011],
    # DSGE solution types
    :PerturbationSolution => [:schmitt_grohe_uribe2004, :kim_kim_schaumburg_sims2008],
    :ProjectionSolution => [:judd1998, :judd_maliar_maliar_valero2014],
    :LinearDSGE => [:sims2002],
    # Bayesian DSGE
    :BayesianDSGE => [:herbst_schorfheide2015, :herbst_schorfheide2014, :an_schorfheide2007],
    :estimate_dsge_bayes => [:herbst_schorfheide2015, :herbst_schorfheide2014, :an_schorfheide2007],
    # Analytical moments
    :analytical_moments => [:hamilton1994, :fernandez_villaverde_rubio_schorfheide2016],
    :solve_lyapunov => [:hamilton1994, :fernandez_villaverde_rubio_schorfheide2016],
    # DiD / Event Study
    :DIDResult => [:callaway_santanna2021, :goodman_bacon2021],
    :EventStudyLP => [:jorda2005, :dube_girardi_jorda_taylor2023],
    :LPDiDResult => [:dube_girardi_jorda_taylor2023, :jorda2005],
    :BaconDecomposition => [:goodman_bacon2021],
    :PretrendTestResult => [:callaway_santanna2021],
    :NegativeWeightResult => [:dechaisemartin_dhaultfoeuille2020],
    :HonestDiDResult => [:rambachan_roth2023, :armstrong_kolesar2018],
    :callaway_santanna => [:callaway_santanna2021],
    :twfe => [:goodman_bacon2021],
    :sun_abraham => [:sun_abraham2021],
    :bjs => [:borusyak_jaravel_spiess2024],
    :did_multiplegt => [:dechaisemartin_dhaultfoeuille2020],
    :lp_did => [:dube_girardi_jorda_taylor2023, :jorda2005],
    # Structural break tests
    :AndrewsResult => [:andrews1993, :andrews_ploberger1994, :hansen1997],
    :BaiPerronResult => [:bai_perron1998, :bai_perron2003],
    :andrews_test => [:andrews1993, :andrews_ploberger1994, :hansen1997],
    :bai_perron_test => [:bai_perron1998, :bai_perron2003],
    # Panel unit root tests
    :PANICResult => [:bai_ng2004, :bai_ng2010],
    :PesaranCIPSResult => [:pesaran2007],
    :MoonPerronResult => [:moon_perron2004],
    :panic_test => [:bai_ng2004, :bai_ng2010],
    :pesaran_cips_test => [:pesaran2007],
    :moon_perron_test => [:moon_perron2004],
    # First-generation panel unit root tests (EV-20, #428)
    :LLCResult => [:levin_lin_chu2002],
    :IPSResult => [:im_pesaran_shin2003],
    :BreitungPanelResult => [:breitung2000],
    :FisherPanelResult => [:maddala_wu1999, :choi2001],
    :HadriResult => [:hadri2000],
    :llc_test => [:levin_lin_chu2002],
    :ips_test => [:im_pesaran_shin2003],
    :breitung_panel_test => [:breitung2000],
    :fisher_panel_test => [:maddala_wu1999, :choi2001],
    :hadri_test => [:hadri2000],
    # Panel cointegration tests (EV-21, #429)
    :PedroniResult => [:pedroni1999, :pedroni2004],
    :KaoResult => [:kao1999],
    :WesterlundResult => [:westerlund2007, :persyn_westerlund2008],
    :FisherJohansenResult => [:maddala_wu1999, :choi2001, :johansen1991],
    :pedroni_test => [:pedroni1999, :pedroni2004],
    :kao_test => [:kao1999],
    :westerlund_test => [:westerlund2007, :persyn_westerlund2008],
    :fisher_johansen_test => [:maddala_wu1999, :choi2001, :johansen1991],
    # Dumitrescu-Hurlin panel Granger non-causality (EV-24, #432)
    :DumitrescuHurlinResult => [:dumitrescu_hurlin2012, :granger1969],
    :dh_causality_test => [:dumitrescu_hurlin2012, :granger1969],
    # EDF goodness-of-fit battery (EV-26, #434)
    :EDFTestResult => [:anderson_darling1954, :stephens1974, :lilliefors1967,
                       :dallal_wilkinson1986, :marsaglia_tsang_wang2003],
    :edf_test => [:anderson_darling1954, :stephens1974, :lilliefors1967,
                  :dallal_wilkinson1986, :marsaglia_tsang_wang2003],
    # Factor model break tests
    :FactorBreakResult => [:breitung_eickmeier2011, :chen_dolado_gonzalo2014, :han_inoue2015],
    :factor_break_test => [:breitung_eickmeier2011, :chen_dolado_gonzalo2014, :han_inoue2015],
    :breitung_eickmeier => [:breitung_eickmeier2011],
    :chen_dolado_gonzalo => [:chen_dolado_gonzalo2014],
    :han_inoue => [:han_inoue2015],
    # Data sources (symbol dispatch)
    :fred_md => [:mccracken_ng2016],
    :fred_qd => [:mccracken_ng2020],
    :pwt => [:feenstra_etal2015],
    # Cross-sectional models
    :RegModel => [:wooldridge2010, :white1980],
    :LogitModel => [:wooldridge2010, :cameron_trivedi2005, :mcfadden1974],
    :ProbitModel => [:wooldridge2010, :cameron_trivedi2005],
    :MarginalEffects => [:cameron_trivedi2005],
    # OLS residual diagnostics (EV-31, #439)
    :RegDiagnosticResult => [:white1980, :breusch_pagan1979, :koenker1981,
                             :glejser1969, :harvey1976, :godfrey1978, :ramsey1969,
                             :chow1960],
    # Stability & influence diagnostics (EV-32, #440)
    :StabilityResult => [:brown_durbin_evans1975, :edgerton_wells1994],
    :InfluenceStats => [:belsley_kuh_welsch1980, :cook1977],
    :recursive_residuals => [:brown_durbin_evans1975],
    :cusum_test => [:brown_durbin_evans1975],
    :cusumsq_test => [:brown_durbin_evans1975, :edgerton_wells1994],
    :chow_test => [:chow1960],
    :influence_stats => [:belsley_kuh_welsch1980, :cook1977],
    :estimate_reg => [:wooldridge2010, :white1980],
    :estimate_iv => [:wooldridge2010, :staiger_stock1997],
    :estimate_logit => [:wooldridge2010, :cameron_trivedi2005, :mcfadden1974],
    :estimate_probit => [:wooldridge2010, :cameron_trivedi2005],
    # Penalized regression (EV-03, #411)
    :PenalizedRegModel => [:tibshirani1996, :zou_hastie2005, :friedman2010,
                           :hoerl_kennard1970, :zou2006, :belloni_chernozhukov2013],
    :estimate_lasso => [:tibshirani1996, :friedman2010],
    :estimate_ridge => [:hoerl_kennard1970, :friedman2010],
    :estimate_elastic_net => [:zou_hastie2005, :friedman2010, :zou2006],
    # Variable selection — stepwise / best-subset / GETS (EV-04, #412)
    :SelectionResult => [:hoover_perez1999, :hendry_krolzig2005, :pretis2018],
    :select_variables => [:hoover_perez1999, :hendry_krolzig2005, :pretis2018],
    # Censored / truncated regression (EV-17, #425)
    :TobitModel => [:tobin1958, :olsen1978, :mcdonald_moffitt1980, :wooldridge2010],
    :TruncRegModel => [:hausman_wise1977, :wooldridge2010],
    :estimate_tobit => [:tobin1958, :olsen1978, :mcdonald_moffitt1980],
    :estimate_truncreg => [:hausman_wise1977, :wooldridge2010],
    # Heckman sample-selection model (EV-18, #426)
    :HeckmanModel => [:heckman1979, :greene2018, :mroz1987, :wooldridge2010],
    :estimate_heckman => [:heckman1979, :greene2018, :wooldridge2010],
    :RobustRegModel => [:huber1964, :yohai1987, :salibian_yohai2006, :huber_ronchetti2009, :brownlee1965],  # EV-40 (#448)
    :estimate_robust => [:huber1964, :yohai1987, :salibian_yohai2006, :huber_ronchetti2009],  # EV-40 (#448)
    # Single-equation cointegrating regression (EV-10, #418)
    :CointRegModel => [:phillips_hansen1990, :park1992, :saikkonen1991, :stock_watson1993],
    :PanelCointRegModel => [:pedroni2000, :pedroni2001, :kao_chiang2000, :mark_sul2003],  # EV-22 (#430)
    :estimate_cointreg => [:phillips_hansen1990, :park1992, :saikkonen1991, :stock_watson1993],
    # Residual-based / parameter-stability cointegration tests (EV-11)
    :EngleGrangerResult => [:engle_granger1987, :mackinnon2010],
    :engle_granger_test => [:engle_granger1987, :mackinnon2010],
    :PhillipsOuliarisResult => [:phillips_ouliaris1990, :mackinnon2010],
    :phillips_ouliaris_test => [:phillips_ouliaris1990, :mackinnon2010],
    :HansenInstabilityResult => [:hansen1992_instability],
    :hansen_instability_test => [:hansen1992_instability],
    :ParkAddedResult => [:park1990_added, :park1992],
    :park_added_test => [:park1990_added, :park1992],
    # Ordered & Multinomial models
    :OrderedLogitModel => [:mccullagh1980, :brant1990, :wooldridge2010],
    :OrderedProbitModel => [:mccullagh1980, :wooldridge2010],
    :MultinomialLogitModel => [:mcfadden1974, :hausman_mcfadden1984, :wooldridge2010],
    :estimate_ologit => [:mccullagh1980, :brant1990, :wooldridge2010],
    :estimate_oprobit => [:mccullagh1980, :wooldridge2010],
    :estimate_mlogit => [:mcfadden1974, :hausman_mcfadden1984, :wooldridge2010],
    # Panel regression
    :PanelRegModel => [:baltagi2021, :wooldridge2010],
    :PanelIVModel => [:baltagi2021, :wooldridge2010, :hausman_taylor1981],
    :PanelLogitModel => [:chamberlain1980, :wooldridge2010],
    :PanelProbitModel => [:wooldridge2010],
    :PanelTestResult => [:baltagi2021],
    :panel_fe => [:baltagi2021, :wooldridge2010],
    :panel_re => [:baltagi2021, :wooldridge2010, :mundlak1978],
    :panel_fd => [:baltagi2021, :wooldridge2010],
    :panel_between => [:baltagi2021, :wooldridge2010],
    :panel_cre => [:baltagi2021, :mundlak1978],
    :panel_iv => [:baltagi2021, :wooldridge2010, :hausman_taylor1981],
    :panel_logit => [:chamberlain1980, :wooldridge2010],
    :panel_probit => [:wooldridge2010],
)

# ICA method → additional ref keys (appended to ICASVARResult base refs)
const _ICA_METHOD_REFS = Dict{Symbol, Vector{Symbol}}(
    :fastica => [:hyvarinen1999],
    :jade => [:cardoso_souloumiac1993],
    :sobi => [:belouchrani1997],
    :dcov => [:szekely_rizzo_bakirov2007, :matteson_tsay2017],
    :hsic => [:gretton2005],
)

# ML distribution → additional ref keys
const _ML_DIST_REFS = Dict{Symbol, Vector{Symbol}}(
    :student_t => Symbol[],
    :mixture_normal => Symbol[],
    :pml => Symbol[],
    :skew_normal => Symbol[],
)

# =============================================================================
# Format Functions
# =============================================================================

function _delatex(s::String)
    out = s
    out = replace(out, "\\\"u" => "\u00fc")  # ü
    out = replace(out, "\\\"a" => "\u00e4")  # ä
    out = replace(out, "\\\"o" => "\u00f6")  # ö
    out = replace(out, "\\\"A" => "\u00c4")  # Ä
    out = replace(out, "\\\\'\\i" => "\u00ed")  # í  (for Antolín-Díaz)
    out = replace(out, "\\\\'i" => "\u00ed")   # í
    out = replace(out, "\\`a" => "\u00e0")    # à
    out = replace(out, "{\\'e}" => "\u00e9")  # é
    out = replace(out, "\\'e" => "\u00e9")    # é
    out = replace(out, "{\\o}" => "\u00f8")   # ø
    out = replace(out, "\\o" => "\u00f8")     # ø
    out = replace(out, "{\\c{c}}" => "\u00e7")  # ç
    out = replace(out, "\\&" => "&")
    out = replace(out, "---" => "\u2014")     # em-dash
    out = replace(out, "--" => "\u2013")      # en-dash
    out = replace(out, r"\{|\}" => "")        # strip remaining braces
    out
end

function _format_ref_text(io::IO, r::_RefEntry)
    a = _delatex(r.authors)
    t = _delatex(r.title)
    if r.entry_type == :book
        println(io, "$a $(r.year). $t. $(r.publisher).")
        !isempty(r.isbn) && println(io, "  ISBN: $(r.isbn)")
    else
        j = _delatex(r.journal)
        vol_str = r.volume
        !isempty(r.issue) && (vol_str *= " ($(r.issue))")
        pages = _delatex(r.pages)
        println(io, "$a $(r.year). \"$t.\" $j $vol_str: $pages.")
        !isempty(r.doi) && println(io, "  DOI: https://doi.org/$(r.doi)")
    end
end

function _format_ref_latex(io::IO, r::_RefEntry)
    key = r.key
    if r.entry_type == :book
        println(io, "\\bibitem{$key} $(r.authors). $(r.year). \\textit{$(r.title)}. $(r.publisher).",
            !isempty(r.isbn) ? " ISBN: $(r.isbn)." : "")
    else
        vol_str = r.volume
        !isempty(r.issue) && (vol_str *= " ($(r.issue))")
        pages = r.pages
        doi_str = !isempty(r.doi) ? " \\url{https://doi.org/$(r.doi)}" : ""
        println(io, "\\bibitem{$key} $(r.authors). $(r.year). ``$(r.title).'' \\textit{$(r.journal)} $vol_str: $pages.$doi_str")
    end
end

function _format_ref_bibtex(io::IO, r::_RefEntry)
    key = r.key
    if r.entry_type == :book
        println(io, "@book{$key,")
        println(io, "  author    = {$(r.authors)},")
        println(io, "  title     = {$(r.title)},")
        println(io, "  year      = {$(r.year)},")
        println(io, "  publisher = {$(r.publisher)},")
        !isempty(r.isbn) && println(io, "  isbn      = {$(r.isbn)},")
        !isempty(r.doi) && println(io, "  doi       = {$(r.doi)},")
        println(io, "}")
    else
        etype = r.entry_type == :incollection ? "incollection" : "article"
        println(io, "@$etype{$key,")
        println(io, "  author  = {$(r.authors)},")
        println(io, "  title   = {$(r.title)},")
        btype = r.entry_type == :incollection ? "booktitle" : "journal"
        println(io, "  $btype = {$(r.journal)},")
        println(io, "  year    = {$(r.year)},")
        !isempty(r.volume) && println(io, "  volume  = {$(r.volume)},")
        !isempty(r.issue) && println(io, "  number  = {$(r.issue)},")
        !isempty(r.pages) && println(io, "  pages   = {$(r.pages)},")
        !isempty(r.doi) && println(io, "  doi     = {$(r.doi)},")
        println(io, "}")
    end
end

function _format_ref_html(io::IO, r::_RefEntry)
    a = _delatex(r.authors)
    t = _delatex(r.title)
    if r.entry_type == :book
        doi_link = !isempty(r.isbn) ? " ISBN: $(r.isbn)." : ""
        println(io, "<p>$a $(r.year). <em>$t</em>. $(r.publisher).$doi_link</p>")
    else
        j = _delatex(r.journal)
        vol_str = r.volume
        !isempty(r.issue) && (vol_str *= " ($(r.issue))")
        pages = _delatex(r.pages)
        doi_link = !isempty(r.doi) ? " <a href=\"https://doi.org/$(r.doi)\">DOI</a>" : ""
        println(io, "<p>$a $(r.year). &ldquo;$t.&rdquo; <em>$j</em> $vol_str: $pages.$doi_link</p>")
    end
end

function _format_ref(io::IO, r::_RefEntry, format::Symbol)
    if format == :text
        _format_ref_text(io, r)
    elseif format == :latex
        _format_ref_latex(io, r)
    elseif format == :bibtex
        _format_ref_bibtex(io, r)
    elseif format == :html
        _format_ref_html(io, r)
    else
        throw(ArgumentError("Unknown format: $format. Use :text, :latex, :bibtex, or :html."))
    end
end

# =============================================================================
# Public refs() Methods
# =============================================================================

"""
    refs([io::IO], x; format=get_display_backend())

Print bibliographic references for a model, result, or method.

Supports four output formats via the `format` keyword:
- `:text` — AEA plain text (default, follows `get_display_backend()`)
- `:latex` — `\\bibitem{}` entries
- `:bibtex` — BibTeX `@article{}`/`@book{}` entries
- `:html` — HTML with clickable DOI links

# Dispatch
- **Instance dispatch**: `refs(model)` prints references for the model type
- **Symbol dispatch**: `refs(:fastica)` prints references for a method name

# Examples
```julia
model = estimate_var(Y, 2)
refs(model)                        # AEA text to stdout
refs(model; format=:bibtex)        # BibTeX entries

refs(:johansen)                    # Johansen (1991)
refs(:fastica; format=:latex)      # Hyvärinen (1999) as \\bibitem
```
"""
function refs(io::IO, keys::Vector{Symbol}; format::Symbol=get_display_backend())
    format = format == :bibtex ? :bibtex : format  # :bibtex is extra, not in display backend
    for k in keys
        haskey(_REFERENCES, k) || throw(ArgumentError("Unknown reference key: $k"))
        _format_ref(io, _REFERENCES[k], format)
    end
end

# --- Symbol dispatch ---
function refs(io::IO, method::Symbol; format::Symbol=get_display_backend())
    haskey(_TYPE_REFS, method) || throw(ArgumentError("Unknown method/type: $method"))
    refs(io, _TYPE_REFS[method]; format=format)
end

# --- Instance dispatch: use type name to look up refs ---
function _refs_for_type(io::IO, x; format::Symbol=get_display_backend())
    tname = Symbol(nameof(typeof(x)))
    haskey(_TYPE_REFS, tname) || throw(ArgumentError("No references available for type: $tname"))
    refs(io, _TYPE_REFS[tname]; format=format)
end

# VAR types
refs(io::IO, ::VARModel; kw...) = refs(io, _TYPE_REFS[:VARModel]; kw...)
refs(io::IO, ::ImpulseResponse; kw...) = refs(io, _TYPE_REFS[:ImpulseResponse]; kw...)
refs(io::IO, ::BayesianImpulseResponse; kw...) = refs(io, _TYPE_REFS[:BayesianImpulseResponse]; kw...)
refs(io::IO, ::FEVD; kw...) = refs(io, _TYPE_REFS[:FEVD]; kw...)
refs(io::IO, ::BayesianFEVD; kw...) = refs(io, _TYPE_REFS[:BayesianFEVD]; kw...)
refs(io::IO, ::HistoricalDecomposition; kw...) = refs(io, _TYPE_REFS[:HistoricalDecomposition]; kw...)
refs(io::IO, ::BayesianHistoricalDecomposition; kw...) = refs(io, _TYPE_REFS[:BayesianHistoricalDecomposition]; kw...)
refs(io::IO, ::AriasSVARResult; kw...) = refs(io, _TYPE_REFS[:AriasSVARResult]; kw...)
refs(io::IO, ::UhligSVARResult; kw...) = refs(io, _TYPE_REFS[:UhligSVARResult]; kw...)
refs(io::IO, ::SVARRestrictions; kw...) = refs(io, _TYPE_REFS[:SVARRestrictions]; kw...)
refs(io::IO, ::SignIdentifiedSet; kw...) = refs(io, _TYPE_REFS[:SignIdentifiedSet]; kw...)
refs(io::IO, ::MinnesotaHyperparameters; kw...) = refs(io, _TYPE_REFS[:MinnesotaHyperparameters]; kw...)
refs(io::IO, ::BVARPosterior; kw...) = refs(io, _TYPE_REFS[:BVARPosterior]; kw...)

# Input-Output analysis
refs(io::IO, ::IOData; kw...) = refs(io, _TYPE_REFS[:IOData]; kw...)
refs(io::IO, ::LeontiefModel; kw...) = refs(io, _TYPE_REFS[:LeontiefModel]; kw...)
refs(io::IO, ::GhoshModel; kw...) = refs(io, _TYPE_REFS[:GhoshModel]; kw...)
refs(io::IO, ::IOMultipliers; kw...) = refs(io, _TYPE_REFS[:IOMultipliers]; kw...)
refs(io::IO, ::LinkageResult; kw...) = refs(io, _TYPE_REFS[:LinkageResult]; kw...)
refs(io::IO, ::SDAResult; kw...) = refs(io, _TYPE_REFS[:SDAResult]; kw...)
refs(io::IO, ::ExtractionResult; kw...) = refs(io, _TYPE_REFS[:ExtractionResult]; kw...)
refs(io::IO, ::FootprintResult; kw...) = refs(io, _TYPE_REFS[:FootprintResult]; kw...)
refs(io::IO, ::BaqaeeFarhiResult; kw...) = refs(io, _TYPE_REFS[:BaqaeeFarhiResult]; kw...)

# LP types
refs(io::IO, ::LPModel; kw...) = refs(io, _TYPE_REFS[:LPModel]; kw...)
refs(io::IO, ::LPImpulseResponse; kw...) = refs(io, _TYPE_REFS[:LPImpulseResponse]; kw...)
refs(io::IO, ::LPIVModel; kw...) = refs(io, _TYPE_REFS[:LPIVModel]; kw...)
refs(io::IO, ::SmoothLPModel; kw...) = refs(io, _TYPE_REFS[:SmoothLPModel]; kw...)
refs(io::IO, ::StateLPModel; kw...) = refs(io, _TYPE_REFS[:StateLPModel]; kw...)
refs(io::IO, ::PropensityLPModel; kw...) = refs(io, _TYPE_REFS[:PropensityLPModel]; kw...)
refs(io::IO, ::StructuralLP; kw...) = refs(io, _TYPE_REFS[:StructuralLP]; kw...)
refs(io::IO, ::LPForecast; kw...) = refs(io, _TYPE_REFS[:LPForecast]; kw...)
refs(io::IO, ::LPFEVD; kw...) = refs(io, _TYPE_REFS[:LPFEVD]; kw...)

# Factor models
refs(io::IO, ::FactorModel; kw...) = refs(io, _TYPE_REFS[:FactorModel]; kw...)
refs(io::IO, ::DynamicFactorModel; kw...) = refs(io, _TYPE_REFS[:DynamicFactorModel]; kw...)
refs(io::IO, ::GeneralizedDynamicFactorModel; kw...) = refs(io, _TYPE_REFS[:GeneralizedDynamicFactorModel]; kw...)
refs(io::IO, ::FactorForecast; kw...) = refs(io, _TYPE_REFS[:FactorForecast]; kw...)

# FAVAR & Structural DFM
refs(io::IO, ::FAVARModel; kw...) = refs(io, _TYPE_REFS[:FAVARModel]; kw...)
refs(io::IO, ::BayesianFAVAR; kw...) = refs(io, _TYPE_REFS[:BayesianFAVAR]; kw...)
refs(io::IO, ::StructuralDFM; kw...) = refs(io, _TYPE_REFS[:StructuralDFM]; kw...)

# Unit root tests
refs(io::IO, ::ADFResult; kw...) = refs(io, _TYPE_REFS[:ADFResult]; kw...)
refs(io::IO, ::KPSSResult; kw...) = refs(io, _TYPE_REFS[:KPSSResult]; kw...)
refs(io::IO, ::PPResult; kw...) = refs(io, _TYPE_REFS[:PPResult]; kw...)
refs(io::IO, ::ZAResult; kw...) = refs(io, _TYPE_REFS[:ZAResult]; kw...)
refs(io::IO, ::NgPerronResult; kw...) = refs(io, _TYPE_REFS[:NgPerronResult]; kw...)
refs(io::IO, ::JohansenResult; kw...) = refs(io, _TYPE_REFS[:JohansenResult]; kw...)

# VECM
refs(io::IO, ::VECMModel; kw...) = refs(io, _TYPE_REFS[:VECMModel]; kw...)
refs(io::IO, ::VECMForecast; kw...) = refs(io, _TYPE_REFS[:VECMForecast]; kw...)
refs(io::IO, ::VECMGrangerResult; kw...) = refs(io, _TYPE_REFS[:VECMGrangerResult]; kw...)

# ARIMA
refs(io::IO, ::ARModel; kw...) = refs(io, _TYPE_REFS[:ARModel]; kw...)
refs(io::IO, ::MAModel; kw...) = refs(io, _TYPE_REFS[:MAModel]; kw...)
refs(io::IO, ::ARMAModel; kw...) = refs(io, _TYPE_REFS[:ARMAModel]; kw...)
refs(io::IO, ::ARIMAModel; kw...) = refs(io, _TYPE_REFS[:ARIMAModel]; kw...)
refs(io::IO, ::ARIMAForecast; kw...) = refs(io, _TYPE_REFS[:ARIMAForecast]; kw...)
refs(io::IO, ::ARIMAOrderSelection; kw...) = refs(io, _TYPE_REFS[:ARIMAOrderSelection]; kw...)

# ARFIMA / long memory (EV-13)
refs(io::IO, ::ARFIMAModel; kw...) = refs(io, _TYPE_REFS[:ARFIMAModel]; kw...)
refs(io::IO, ::GPHResult; kw...) = refs(io, _TYPE_REFS[:GPHResult]; kw...)
refs(io::IO, ::LocalWhittleResult; kw...) = refs(io, _TYPE_REFS[:LocalWhittleResult]; kw...)

# MIDAS (EV-01)
refs(io::IO, ::MidasModel; kw...) = refs(io, _TYPE_REFS[:MidasModel]; kw...)
refs(io::IO, ::MidasForecast; kw...) = refs(io, _TYPE_REFS[:MidasForecast]; kw...)

# ARDL & bounds test (EV-08, #416)
refs(io::IO, ::ARDLModel; kw...) = refs(io, _TYPE_REFS[:ARDLModel]; kw...)
refs(io::IO, ::ARDLBoundsTest; kw...) = refs(io, _TYPE_REFS[:ARDLBoundsTest]; kw...)
# NARDL — nonlinear ARDL (EV-09, #417)
refs(io::IO, ::NARDLModel; kw...) = refs(io, _TYPE_REFS[:NARDLModel]; kw...)
refs(io::IO, ::NARDLSymmetryTest; kw...) = refs(io, _TYPE_REFS[:NARDLSymmetryTest]; kw...)
refs(io::IO, ::NARDLMultipliers; kw...) = refs(io, _TYPE_REFS[:NARDLMultipliers]; kw...)

# Panel ARDL: PMG / MG / DFE (EV-23, #431)
refs(io::IO, ::PMGModel; kw...) = refs(io, _TYPE_REFS[:PMGModel]; kw...)

# GMM
refs(io::IO, ::GMMModel; kw...) = refs(io, _TYPE_REFS[:GMMModel]; kw...)

# SMM
refs(io::IO, ::SMMModel; kw...) = refs(io, _TYPE_REFS[:SMMModel]; kw...)

# Volatility models
refs(io::IO, ::ARCHModel; kw...) = refs(io, _TYPE_REFS[:ARCHModel]; kw...)
refs(io::IO, ::GARCHModel; kw...) = refs(io, _TYPE_REFS[:GARCHModel]; kw...)
refs(io::IO, ::EGARCHModel; kw...) = refs(io, _TYPE_REFS[:EGARCHModel]; kw...)
refs(io::IO, ::GJRGARCHModel; kw...) = refs(io, _TYPE_REFS[:GJRGARCHModel]; kw...)
refs(io::IO, ::GarchMidasModel; kw...) = refs(io, _TYPE_REFS[:GarchMidasModel]; kw...)
refs(io::IO, ::FIGARCHModel; kw...) = refs(io, _TYPE_REFS[:FIGARCHModel]; kw...)
refs(io::IO, ::FIEGARCHModel; kw...) = refs(io, _TYPE_REFS[:FIEGARCHModel]; kw...)
refs(io::IO, ::IGARCHModel; kw...) = refs(io, _TYPE_REFS[:IGARCHModel]; kw...)
refs(io::IO, ::CGARCHModel; kw...) = refs(io, _TYPE_REFS[:CGARCHModel]; kw...)
refs(io::IO, ::APARCHModel; kw...) = refs(io, _TYPE_REFS[:APARCHModel]; kw...)
refs(io::IO, ::MGARCHModel; kw...) = refs(io, _TYPE_REFS[:MGARCHModel]; kw...)   # EV-16 (#424)
refs(io::IO, ::SVModel; kw...) = refs(io, _TYPE_REFS[:SVModel]; kw...)
refs(io::IO, ::VolatilityForecast; kw...) = refs(io, _TYPE_REFS[:VolatilityForecast]; kw...)

# Nonlinear time series (threshold/SETAR)
refs(io::IO, ::ThresholdModel; kw...) = refs(io, _TYPE_REFS[:ThresholdModel]; kw...)
refs(io::IO, ::HansenLinearityTest; kw...) = refs(io, _TYPE_REFS[:HansenLinearityTest]; kw...)
refs(io::IO, ::STARModel; kw...) = refs(io, _TYPE_REFS[:STARModel]; kw...)
refs(io::IO, ::MSRegModel; kw...) = refs(io, _TYPE_REFS[:MSRegModel]; kw...)

# Covariance estimators
refs(io::IO, ::NeweyWestEstimator; kw...) = refs(io, _TYPE_REFS[:NeweyWestEstimator]; kw...)
refs(io::IO, ::WhiteEstimator; kw...) = refs(io, _TYPE_REFS[:WhiteEstimator]; kw...)

# Normality tests
refs(io::IO, ::NormalityTestResult; kw...) = refs(io, _TYPE_REFS[:NormalityTestResult]; kw...)
refs(io::IO, ::NormalityTestSuite; kw...) = refs(io, _TYPE_REFS[:NormalityTestSuite]; kw...)

# Non-Gaussian types with variant-dependent refs
function refs(io::IO, r::ICASVARResult; format::Symbol=get_display_backend())
    base = _TYPE_REFS[:ICASVARResult]
    extra = get(_ICA_METHOD_REFS, r.method, Symbol[])
    refs(io, unique(vcat(base, extra)); format=format)
end

function refs(io::IO, r::NonGaussianMLResult; format::Symbol=get_display_backend())
    base = _TYPE_REFS[:NonGaussianMLResult]
    extra = get(_ML_DIST_REFS, r.distribution, Symbol[])
    refs(io, unique(vcat(base, extra)); format=format)
end

# Heteroskedastic types (concrete type dispatch, no method field)
refs(io::IO, ::MarkovSwitchingSVARResult; kw...) = refs(io, _TYPE_REFS[:MarkovSwitchingSVARResult]; kw...)
refs(io::IO, ::GARCHSVARResult; kw...) = refs(io, _TYPE_REFS[:GARCHSVARResult]; kw...)
refs(io::IO, ::SmoothTransitionSVARResult; kw...) = refs(io, _TYPE_REFS[:SmoothTransitionSVARResult]; kw...)
refs(io::IO, ::ExternalVolatilitySVARResult; kw...) = refs(io, _TYPE_REFS[:ExternalVolatilitySVARResult]; kw...)

# Identifiability test result
refs(io::IO, ::IdentifiabilityTestResult; kw...) = refs(io, [:lanne_meitz_saikkonen2017]; kw...)

# Time series filters
refs(io::IO, ::HPFilterResult; kw...) = refs(io, _TYPE_REFS[:HPFilterResult]; kw...)
refs(io::IO, ::HamiltonFilterResult; kw...) = refs(io, _TYPE_REFS[:HamiltonFilterResult]; kw...)
refs(io::IO, ::BeveridgeNelsonResult; kw...) = refs(io, _TYPE_REFS[:BeveridgeNelsonResult]; kw...)
refs(io::IO, ::BaxterKingResult; kw...) = refs(io, _TYPE_REFS[:BaxterKingResult]; kw...)
refs(io::IO, ::BoostedHPResult; kw...) = refs(io, _TYPE_REFS[:BoostedHPResult]; kw...)
refs(io::IO, ::X13FilterResult; kw...) = refs(io, _TYPE_REFS[:X13FilterResult]; kw...)

# Model comparison tests
refs(io::IO, ::LRTestResult; kw...) = refs(io, _TYPE_REFS[:LRTestResult]; kw...)
refs(io::IO, ::LMTestResult; kw...) = refs(io, _TYPE_REFS[:LMTestResult]; kw...)

# Granger causality
refs(io::IO, ::GrangerCausalityResult; kw...) = refs(io, _TYPE_REFS[:GrangerCausalityResult]; kw...)

# Panel VAR
refs(io::IO, ::PVARModel; kw...) = refs(io, _TYPE_REFS[:PVARModel]; kw...)
refs(io::IO, ::PVARStability; kw...) = refs(io, _TYPE_REFS[:PVARStability]; kw...)
refs(io::IO, ::PVARTestResult; kw...) = refs(io, _TYPE_REFS[:PVARTestResult]; kw...)

# DiD / Event Study
refs(io::IO, ::DIDResult; kw...) = refs(io, _TYPE_REFS[:DIDResult]; kw...)
refs(io::IO, ::EventStudyLP; kw...) = refs(io, _TYPE_REFS[:EventStudyLP]; kw...)
refs(io::IO, ::LPDiDResult; kw...) = refs(io, _TYPE_REFS[:LPDiDResult]; kw...)
refs(io::IO, ::BaconDecomposition; kw...) = refs(io, _TYPE_REFS[:BaconDecomposition]; kw...)
refs(io::IO, ::PretrendTestResult; kw...) = refs(io, _TYPE_REFS[:PretrendTestResult]; kw...)
refs(io::IO, ::NegativeWeightResult; kw...) = refs(io, _TYPE_REFS[:NegativeWeightResult]; kw...)
refs(io::IO, ::HonestDiDResult; kw...) = refs(io, _TYPE_REFS[:HonestDiDResult]; kw...)

# Data containers (use source_refs field)
function refs(io::IO, d::AbstractMacroData; format::Symbol=get_display_backend())
    isempty(d.source_refs) && throw(ArgumentError(
        "No source references attached to this data object. Set source_refs at construction or use load_example()."))
    refs(io, d.source_refs; format=format)
end

# Nowcasting types
refs(io::IO, ::NowcastDFM; kw...) = refs(io, _TYPE_REFS[:NowcastDFM]; kw...)
refs(io::IO, ::NowcastBVAR; kw...) = refs(io, _TYPE_REFS[:NowcastBVAR]; kw...)
refs(io::IO, ::NowcastBridge; kw...) = refs(io, _TYPE_REFS[:NowcastBridge]; kw...)
refs(io::IO, ::NowcastResult; kw...) = refs(io, _TYPE_REFS[:NowcastResult]; kw...)
refs(io::IO, ::NowcastNews; kw...) = refs(io, _TYPE_REFS[:NowcastNews]; kw...)

# DSGE types
refs(io::IO, ::DSGESolution; kw...) = refs(io, _TYPE_REFS[:DSGESolution]; kw...)
refs(io::IO, ::DSGEEstimation; kw...) = refs(io, _TYPE_REFS[:DSGEEstimation]; kw...)
refs(io::IO, ::PerfectForesightPath; kw...) = refs(io, _TYPE_REFS[:PerfectForesightPath]; kw...)
refs(io::IO, ::DSGESpec; kw...) = refs(io, _TYPE_REFS[:DSGESpec]; kw...)
refs(io::IO, ::PerturbationSolution; kw...) = refs(io, _TYPE_REFS[:PerturbationSolution]; kw...)
refs(io::IO, ::ProjectionSolution; kw...) = refs(io, _TYPE_REFS[:ProjectionSolution]; kw...)
refs(io::IO, ::LinearDSGE; kw...) = refs(io, _TYPE_REFS[:LinearDSGE]; kw...)

# Bayesian DSGE
refs(io::IO, ::BayesianDSGE; kw...) = refs(io, _TYPE_REFS[:BayesianDSGE]; kw...)

# OccBin types
refs(io::IO, ::OccBinSolution; kw...) = refs(io, _TYPE_REFS[:OccBinSolution]; kw...)
refs(io::IO, ::OccBinIRF; kw...) = refs(io, _TYPE_REFS[:OccBinIRF]; kw...)

# Structural break types
refs(io::IO, ::AndrewsResult; kw...) = refs(io, _TYPE_REFS[:AndrewsResult]; kw...)
refs(io::IO, ::BaiPerronResult; kw...) = refs(io, _TYPE_REFS[:BaiPerronResult]; kw...)

# Panel unit root types
refs(io::IO, ::PANICResult; kw...) = refs(io, _TYPE_REFS[:PANICResult]; kw...)
refs(io::IO, ::PesaranCIPSResult; kw...) = refs(io, _TYPE_REFS[:PesaranCIPSResult]; kw...)
refs(io::IO, ::MoonPerronResult; kw...) = refs(io, _TYPE_REFS[:MoonPerronResult]; kw...)
# First-generation panel unit root tests (EV-20, #428)
refs(io::IO, ::LLCResult; kw...) = refs(io, _TYPE_REFS[:LLCResult]; kw...)
refs(io::IO, ::IPSResult; kw...) = refs(io, _TYPE_REFS[:IPSResult]; kw...)
refs(io::IO, ::BreitungPanelResult; kw...) = refs(io, _TYPE_REFS[:BreitungPanelResult]; kw...)
refs(io::IO, ::FisherPanelResult; kw...) = refs(io, _TYPE_REFS[:FisherPanelResult]; kw...)
refs(io::IO, ::HadriResult; kw...) = refs(io, _TYPE_REFS[:HadriResult]; kw...)
# Panel cointegration tests (EV-21, #429)
refs(io::IO, ::PedroniResult; kw...) = refs(io, _TYPE_REFS[:PedroniResult]; kw...)
refs(io::IO, ::KaoResult; kw...) = refs(io, _TYPE_REFS[:KaoResult]; kw...)
refs(io::IO, ::WesterlundResult; kw...) = refs(io, _TYPE_REFS[:WesterlundResult]; kw...)
refs(io::IO, ::DumitrescuHurlinResult; kw...) = refs(io, _TYPE_REFS[:DumitrescuHurlinResult]; kw...)
# EDF goodness-of-fit battery (EV-26, #434)
refs(io::IO, ::EDFTestResult; kw...) = refs(io, _TYPE_REFS[:EDFTestResult]; kw...)
refs(io::IO, ::FisherJohansenResult; kw...) = refs(io, _TYPE_REFS[:FisherJohansenResult]; kw...)
# Residual-based / parameter-stability cointegration tests (EV-11, #419)
refs(io::IO, ::EngleGrangerResult; kw...) = refs(io, _TYPE_REFS[:EngleGrangerResult]; kw...)
refs(io::IO, ::PhillipsOuliarisResult; kw...) = refs(io, _TYPE_REFS[:PhillipsOuliarisResult]; kw...)
refs(io::IO, ::HansenInstabilityResult; kw...) = refs(io, _TYPE_REFS[:HansenInstabilityResult]; kw...)
refs(io::IO, ::ParkAddedResult; kw...) = refs(io, _TYPE_REFS[:ParkAddedResult]; kw...)

# Factor break types
refs(io::IO, ::FactorBreakResult; kw...) = refs(io, _TYPE_REFS[:FactorBreakResult]; kw...)

# Cross-sectional models
# IV k-class (LIML/Fuller) models additionally cite Anderson-Rubin, Fuller, Bekker (EV-36, #444).
refs(io::IO, m::RegModel; kw...) = refs(io,
    m.method == :iv ?
        (m.kappa_hat !== nothing ?
            [:wooldridge2010, :white1980, :staiger_stock1997, :anderson_rubin1949, :fuller1977, :bekker1994] :
            [:wooldridge2010, :white1980, :staiger_stock1997]) :
        [:wooldridge2010, :white1980]; kw...)
refs(io::IO, ::LogitModel; kw...) = refs(io, _TYPE_REFS[:LogitModel]; kw...)
refs(io::IO, ::ProbitModel; kw...) = refs(io, _TYPE_REFS[:ProbitModel]; kw...)
refs(io::IO, ::MarginalEffects; kw...) = refs(io, _TYPE_REFS[:MarginalEffects]; kw...)
# Stability & influence diagnostics (EV-32, #440)
refs(io::IO, r::StabilityResult; kw...) = refs(io, r.kind == :cusumsq ?
    [:brown_durbin_evans1975, :edgerton_wells1994] : [:brown_durbin_evans1975]; kw...)
refs(io::IO, ::InfluenceStats; kw...) = refs(io, _TYPE_REFS[:InfluenceStats]; kw...)
# Penalized regression (EV-03, #411): reference set depends on the fitted variant.
function refs(io::IO, m::PenalizedRegModel; kw...)
    ks = m.alpha == 1 ? [:tibshirani1996, :friedman2010] :
         m.alpha == 0 ? [:hoerl_kennard1970, :friedman2010] :
                        [:zou_hastie2005, :friedman2010]
    m.adaptive && push!(ks, :zou2006)
    m.post && push!(ks, :belloni_chernozhukov2013)
    refs(io, ks; kw...)
end

# Variable selection — stepwise / best-subset / GETS (EV-04, #412)
refs(io::IO, ::SelectionResult; kw...) = refs(io, _TYPE_REFS[:SelectionResult]; kw...)

# Censored / truncated regression (EV-17, #425)
refs(io::IO, ::TobitModel; kw...) = refs(io, _TYPE_REFS[:TobitModel]; kw...)
refs(io::IO, ::TruncRegModel; kw...) = refs(io, _TYPE_REFS[:TruncRegModel]; kw...)
refs(io::IO, ::HeckmanModel; kw...) = refs(io, _TYPE_REFS[:HeckmanModel]; kw...)  # EV-18 (#426)
refs(io::IO, ::RobustRegModel; kw...) = refs(io, _TYPE_REFS[:RobustRegModel]; kw...)  # EV-40 (#448)
refs(io::IO, ::CointRegModel; kw...) = refs(io, _TYPE_REFS[:CointRegModel]; kw...)  # EV-10 (#418)
refs(io::IO, ::PanelCointRegModel; kw...) = refs(io, _TYPE_REFS[:PanelCointRegModel]; kw...)  # EV-22 (#430)

# Ordered & Multinomial models
refs(io::IO, ::OrderedLogitModel; kw...) = refs(io, _TYPE_REFS[:OrderedLogitModel]; kw...)
refs(io::IO, ::OrderedProbitModel; kw...) = refs(io, _TYPE_REFS[:OrderedProbitModel]; kw...)
refs(io::IO, ::MultinomialLogitModel; kw...) = refs(io, _TYPE_REFS[:MultinomialLogitModel]; kw...)

# Panel regression
function refs(io::IO, m::PanelRegModel; format::Symbol=get_display_backend())
    base = [:baltagi2021, :wooldridge2010]
    if m.method == :re
        push!(base, :mundlak1978)
    elseif m.method == :cre
        push!(base, :mundlak1978)
    end
    refs(io, unique(base); format=format)
end
function refs(io::IO, m::PanelIVModel; format::Symbol=get_display_backend())
    base = [:baltagi2021, :wooldridge2010]
    if m.method == :hausman_taylor
        push!(base, :hausman_taylor1981)
    end
    refs(io, unique(base); format=format)
end
refs(io::IO, ::PanelLogitModel; kw...) = refs(io, _TYPE_REFS[:PanelLogitModel]; kw...)
refs(io::IO, ::PanelProbitModel; kw...) = refs(io, _TYPE_REFS[:PanelProbitModel]; kw...)
refs(io::IO, ::PanelTestResult; kw...) = refs(io, _TYPE_REFS[:PanelTestResult]; kw...)

# OLS residual diagnostics (EV-31, #439): pick the reference(s) matching the test.
function refs(io::IO, r::RegDiagnosticResult; kw...)
    nm = r.test_name
    ks = startswith(nm, "White")            ? [:white1980] :
         startswith(nm, "Breusch-Pagan")    ? [:breusch_pagan1979, :koenker1981] :
         startswith(nm, "Glejser")          ? [:glejser1969] :
         startswith(nm, "Harvey")           ? [:harvey1976] :
         startswith(nm, "Breusch-Godfrey")  ? [:breusch1978, :godfrey1978] :
         startswith(nm, "Ramsey")           ? [:ramsey1969] :
         startswith(nm, "Chow")             ? [:chow1960] :
                                              _TYPE_REFS[:RegDiagnosticResult]
    refs(io, ks; kw...)
end

# Forecast evaluation & combination (EV-39, #447)
refs(io::IO, ::ForecastEvaluation; kw...) = refs(io, _TYPE_REFS[:ForecastEvaluation]; kw...)
refs(io::IO, ::DMTestResult; kw...) = refs(io, _TYPE_REFS[:DMTestResult]; kw...)
refs(io::IO, ::ClarkWestResult; kw...) = refs(io, _TYPE_REFS[:ClarkWestResult]; kw...)
refs(io::IO, ::MincerZarnowitzResult; kw...) = refs(io, _TYPE_REFS[:MincerZarnowitzResult]; kw...)
refs(io::IO, ::ForecastEncompassingResult; kw...) = refs(io, _TYPE_REFS[:ForecastEncompassingResult]; kw...)
refs(io::IO, ::ForecastCombination; kw...) = refs(io, _TYPE_REFS[:ForecastCombination]; kw...)

# --- Convenience: stdout fallback ---
function refs(x; kw...)
    io = IOBuffer()
    refs(io, x; kw...)
    String(take!(io))
end

