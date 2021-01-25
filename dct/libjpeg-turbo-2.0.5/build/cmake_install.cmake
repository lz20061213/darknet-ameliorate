# Install script for directory: /home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/liuzhuang/codes/darknet-ameliorate/dct/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libturbojpeg.so.0.2.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libturbojpeg.so.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libturbojpeg.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "/home/liuzhuang/codes/darknet-ameliorate/dct/install/lib")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/build/libturbojpeg.so.0.2.0"
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/build/libturbojpeg.so.0"
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/build/libturbojpeg.so"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libturbojpeg.so.0.2.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libturbojpeg.so.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libturbojpeg.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
           NEW_RPATH "/home/liuzhuang/codes/darknet-ameliorate/dct/install/lib")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/tjbench" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/tjbench")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/tjbench"
         RPATH "/home/liuzhuang/codes/darknet-ameliorate/dct/install/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/build/tjbench")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/tjbench" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/tjbench")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/tjbench"
         OLD_RPATH "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/build:"
         NEW_RPATH "/home/liuzhuang/codes/darknet-ameliorate/dct/install/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/tjbench")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/build/libturbojpeg.a")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/turbojpeg.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/build/libjpeg.a")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/rdjpgcom" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/rdjpgcom")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/rdjpgcom"
         RPATH "/home/liuzhuang/codes/darknet-ameliorate/dct/install/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/build/rdjpgcom")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/rdjpgcom" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/rdjpgcom")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/rdjpgcom"
         OLD_RPATH "::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
         NEW_RPATH "/home/liuzhuang/codes/darknet-ameliorate/dct/install/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/rdjpgcom")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/wrjpgcom" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/wrjpgcom")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/wrjpgcom"
         RPATH "/home/liuzhuang/codes/darknet-ameliorate/dct/install/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/build/wrjpgcom")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/wrjpgcom" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/wrjpgcom")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/wrjpgcom"
         OLD_RPATH "::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
         NEW_RPATH "/home/liuzhuang/codes/darknet-ameliorate/dct/install/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/wrjpgcom")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/doc/libjpeg-turbo" TYPE FILE FILES
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/README.ijg"
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/README.md"
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/example.txt"
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/tjexample.c"
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/libjpeg.txt"
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/structure.txt"
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/usage.txt"
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/wizard.txt"
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/LICENSE.md"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/man/man1" TYPE FILE FILES
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/cjpeg.1"
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/djpeg.1"
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/jpegtran.1"
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/rdjpgcom.1"
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/wrjpgcom.1"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/build/pkgscripts/libjpeg.pc"
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/build/pkgscripts/libturbojpeg.pc"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/build/jconfig.h"
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/jerror.h"
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/jmorecfg.h"
    "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/jpeglib.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/build/simd/cmake_install.cmake")
  include("/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/build/sharedlib/cmake_install.cmake")
  include("/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/build/md5/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/liuzhuang/codes/darknet-ameliorate/dct/libjpeg-turbo-2.0.5/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
