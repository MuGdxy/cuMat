//Included inside Matrix

////first pull the members from MatrixBase, defined in MatrixBlockPluginRvalue to this scope
////this makes the const versions available here
//using Base::block;
////Apparently, this is not legal C++ code, even if Visual Studio and some versions of GCC allows that (other GCC versions don't)
////So I have to manually provide the const versions

//most general version, static size


