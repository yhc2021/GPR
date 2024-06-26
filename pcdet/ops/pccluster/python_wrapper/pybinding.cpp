#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "clusteringprocessor/clusteringprocessor.h"

namespace py = pybind11;

PYBIND11_MODULE(pccluster, m) {

    m.doc() = "Python Cluster";
    m.attr("__version__") = 1;

    py::class_<depth_clustering::Params>(m, "Parameters")
        .def(py::init<>())
        .def_readwrite("verbose",       &depth_clustering::Params::verbose)
        .def_readwrite("dist_thr",    &depth_clustering::Params::dist_thr)
        .def_readwrite("angle",   &depth_clustering::Params::angle)
        .def_readwrite("min_cluster_size",    &depth_clustering::Params::min_cluster_size)
        .def_readwrite("max_cluster_size",      &depth_clustering::Params::max_cluster_size)
        .def_readwrite("dataset_flag",      &depth_clustering::Params::dataset_flag);

    py::class_<depth_clustering::ClusteringProcessor>(m, "clusteringprocessor")
        .def(py::init<depth_clustering::Params>())
        // .def("EigenToCloud",       &depth_clustering::ClusteringProcessor::EigenToCloud)
        .def("getTimeTakenUs",    &depth_clustering::ClusteringProcessor::getTimeTakenUs)
        .def("getCluster",       &depth_clustering::ClusteringProcessor::getCluster)
        .def("getNoncluster",    &depth_clustering::ClusteringProcessor::getNoncluster)
        // .def("getCenters",      &depth_clustering::ClusteringProcessor::getCenters)
        // .def("getNormals",      &depth_clustering::ClusteringProcessor::getNormals)        
        .def("process_cluster",  &depth_clustering::ClusteringProcessor::process_cluster);
        // .def_readwrite("sensor_height_", &ClusteringProcessor::sensor_height_);
}