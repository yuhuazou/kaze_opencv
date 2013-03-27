#ifndef _PREDEP_H_
#define _PREDEP_H_

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>
#include <iostream>

#include "opencv2/core/version.hpp"

#if ((CV_MAJOR_VERSION>=2) && (CV_MINOR_VERSION>=4)) 
#define CV_VERSION_ID       CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#else
#define CV_VERSION_ID "_minimum_version_2.4.0_please_update_your_OpenCV"
#endif

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif

#endif
