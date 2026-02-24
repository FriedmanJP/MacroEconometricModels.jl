# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

# =============================================================================
# refs() — Multi-Format Bibliographic References
# =============================================================================

const _RefEntry = @NamedTuple{
    key::Symbol, authors::String, year::Int, title::String,
    journal::String, volume::String, issue::String, pages::String,
    doi::String, isbn::String, publisher::String, entry_type::Symbol
}

const _REFERENCES = Dict{Symbol, _RefEntry}(
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
    :glosten_jagannathan_runkle1993 => (key=:glosten_jagannathan_runkle1993,
        authors="Glosten, Lawrence R. and Jagannathan, Ravi and Runkle, David E.", year=1993,
        title="On the Relation Between the Expected Value and the Volatility of the Nominal Excess Return on Stocks",
        journal="Journal of Finance", volume="48", issue="5", pages="1779--1801",
        doi="10.1111/j.1540-6261.1993.tb05128.x", isbn="", publisher="", entry_type=:article),
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
)

# --- Type/method → reference keys mapping ---

const _TYPE_REFS = Dict{Symbol, Vector{Symbol}}(
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
    :SVModel => [:taylor1986, :kim_shephard_chib1998, :omori2007],
    :VolatilityForecast => [:engle1982, :bollerslev1986],
    :arch => [:engle1982],
    :garch => [:bollerslev1986],
    :egarch => [:nelson1991],
    :gjr_garch => [:glosten_jagannathan_runkle1993],
    :sv => [:taylor1986, :kim_shephard_chib1998, :omori2007],
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
    :DSGEEstimation => [:sims2002, :christiano_eichenbaum_evans2005, :hansen_singleton1982],
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
    # Analytical moments
    :analytical_moments => [:hamilton1994, :fernandez_villaverde_rubio_schorfheide2016],
    :solve_lyapunov => [:hamilton1994, :fernandez_villaverde_rubio_schorfheide2016],
    # Data sources (symbol dispatch)
    :fred_md => [:mccracken_ng2016],
    :fred_qd => [:mccracken_ng2020],
    :pwt => [:feenstra_etal2015],
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

# GMM
refs(io::IO, ::GMMModel; kw...) = refs(io, _TYPE_REFS[:GMMModel]; kw...)

# SMM
refs(io::IO, ::SMMModel; kw...) = refs(io, _TYPE_REFS[:SMMModel]; kw...)

# Volatility models
refs(io::IO, ::ARCHModel; kw...) = refs(io, _TYPE_REFS[:ARCHModel]; kw...)
refs(io::IO, ::GARCHModel; kw...) = refs(io, _TYPE_REFS[:GARCHModel]; kw...)
refs(io::IO, ::EGARCHModel; kw...) = refs(io, _TYPE_REFS[:EGARCHModel]; kw...)
refs(io::IO, ::GJRGARCHModel; kw...) = refs(io, _TYPE_REFS[:GJRGARCHModel]; kw...)
refs(io::IO, ::SVModel; kw...) = refs(io, _TYPE_REFS[:SVModel]; kw...)
refs(io::IO, ::VolatilityForecast; kw...) = refs(io, _TYPE_REFS[:VolatilityForecast]; kw...)

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

# Model comparison tests
refs(io::IO, ::LRTestResult; kw...) = refs(io, _TYPE_REFS[:LRTestResult]; kw...)
refs(io::IO, ::LMTestResult; kw...) = refs(io, _TYPE_REFS[:LMTestResult]; kw...)

# Granger causality
refs(io::IO, ::GrangerCausalityResult; kw...) = refs(io, _TYPE_REFS[:GrangerCausalityResult]; kw...)

# Panel VAR
refs(io::IO, ::PVARModel; kw...) = refs(io, _TYPE_REFS[:PVARModel]; kw...)
refs(io::IO, ::PVARStability; kw...) = refs(io, _TYPE_REFS[:PVARStability]; kw...)
refs(io::IO, ::PVARTestResult; kw...) = refs(io, _TYPE_REFS[:PVARTestResult]; kw...)

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

# OccBin types
refs(io::IO, ::OccBinSolution; kw...) = refs(io, _TYPE_REFS[:OccBinSolution]; kw...)
refs(io::IO, ::OccBinIRF; kw...) = refs(io, _TYPE_REFS[:OccBinIRF]; kw...)

# --- Convenience: stdout fallback ---
function refs(x; kw...)
    io = IOBuffer()
    refs(io, x; kw...)
    String(take!(io))
end

