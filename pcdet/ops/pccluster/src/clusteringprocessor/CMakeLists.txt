add_library(clusteringprocessor SHARED clusteringprocessor.cpp)
# target_include_directories(clusteringprocessor PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
# )

target_link_libraries(clusteringprocessor
                      Eigen3::Eigen
                    #   identifiable
                      image_labeler
                      velodyne_utils
                      projections
                      ${MY_QT_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${PCL_LIBRARIES}
                      ${CMAKE_THREAD_LIBS_INIT})
# add_library(CLUSTERING::clusteringprocessor ALIAS clusteringprocessor)

# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
#   DESTINATION include
# )
# install(TARGETS clusteringprocessor
#   EXPORT CLUSTERINGConfig
#   LIBRARY DESTINATION lib
# )

# export(TARGETS clusteringprocessor
#   NAMESPACE CLUSTERING::
#   FILE "${CMAKE_CURRENT_BINARY_DIR}/CLUSTERINGConfig.cmake"
# )
# install(EXPORT CLUSTERINGConfig
#   DESTINATION "${CMAKE_INSTALL_DATADIR}/CLUSTERING/cmake"
#   NAMESPACE CLUSTERING::
# )