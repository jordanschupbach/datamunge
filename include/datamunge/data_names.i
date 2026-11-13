// Names of the datamunge::data dataset functions in SWIG-generated bindings.
//
// Rosetta-generated bindings (Python, R, JavaScript, Lua, Common Lisp) export the namespace as a
// nested `data` submodule (dm.data.iris()) via the "modules" key of their policy.json. SWIG
// flattens namespaces, so these targets keep the grouping as a prefix instead: data_iris(), or
// DataIris() in Go. %include this before the facade header, and add a line here for each new
// function in datamunge::data (include/datamunge/datamunge.hpp).
#ifdef SWIGGO
%rename(DataIris) datamunge::data::iris;
%rename(DataPenguins) datamunge::data::penguins;
%rename(DataToothGrowth) datamunge::data::tooth_growth;
%rename(DataWine) datamunge::data::wine;
%rename(DataTitanic) datamunge::data::titanic;
%rename(DataEightSchools) datamunge::data::eight_schools;
%rename(DataAirPassengers) datamunge::data::air_passengers;
%rename(DataVolcano) datamunge::data::volcano;
%rename(DataFaithful) datamunge::data::faithful;
%rename(DataMtcars) datamunge::data::mtcars;
%rename(DataAirquality) datamunge::data::airquality;
%rename(DataBreastCancer) datamunge::data::breast_cancer;
%rename(DataDiabetes) datamunge::data::diabetes;
%rename(DataDigits) datamunge::data::digits;
%rename(DataDiamonds) datamunge::data::diamonds;
%rename(DataCaliforniaHousing) datamunge::data::california_housing;
%rename(DataLoad) datamunge::data::load;
%rename(DataCatalog) datamunge::data::catalog;
#else
%rename(data_iris) datamunge::data::iris;
%rename(data_penguins) datamunge::data::penguins;
%rename(data_tooth_growth) datamunge::data::tooth_growth;
%rename(data_wine) datamunge::data::wine;
%rename(data_titanic) datamunge::data::titanic;
%rename(data_eight_schools) datamunge::data::eight_schools;
%rename(data_air_passengers) datamunge::data::air_passengers;
%rename(data_volcano) datamunge::data::volcano;
%rename(data_faithful) datamunge::data::faithful;
%rename(data_mtcars) datamunge::data::mtcars;
%rename(data_airquality) datamunge::data::airquality;
%rename(data_breast_cancer) datamunge::data::breast_cancer;
%rename(data_diabetes) datamunge::data::diabetes;
%rename(data_digits) datamunge::data::digits;
%rename(data_diamonds) datamunge::data::diamonds;
%rename(data_california_housing) datamunge::data::california_housing;
%rename(data_load) datamunge::data::load;
%rename(data_catalog) datamunge::data::catalog;
#endif
