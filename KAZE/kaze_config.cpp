
//=============================================================================
//
// Ipoint.cpp
// Author: Pablo F. Alcantarilla
// Institution: University d'Auvergne
// Address: Clermont Ferrand, France
// Date: 21/01/2012
// Email: pablofdezalc@gmail.com
//
// KAZE Features Copyright 2012, Pablo F. Alcantarilla
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file Ipoint.cpp
 * @brief Class that defines a point of interest
 * @date Jan 21, 2012
 * @author Pablo F. Alcantarilla
 */

#include "kaze_config.h"

//*******************************************************************************
//*******************************************************************************

/**
 * @brief Ipoint default constructor
 */
toptions::toptions(void)
{
	soffset = DEFAULT_SCALE_OFFSET;
	omax = DEFAULT_OCTAVE_MAX;
	nsublevels = DEFAULT_NSUBLEVELS;
	dthreshold = DEFAULT_DETECTOR_THRESHOLD;
	dthreshold2 = DEFAULT_DETECTOR_THRESHOLD;
	diffusivity = DEFAULT_DIFFUSIVITY_TYPE;
	descriptor = DEFAULT_DESCRIPTOR_MODE;
	sderivatives = DEFAULT_SIGMA_SMOOTHING_DERIVATIVES;
	upright = DEFAULT_UPRIGHT;
	extended = DEFAULT_EXTENDED;
	save_scale_space = DEFAULT_SAVE_SCALE_SPACE;
	save_keypoints = DEFAULT_SAVE_KEYPOINTS;
	verbosity = DEFAULT_VERBOSITY;
	show_results = DEFAULT_SHOW_RESULTS;
}

//*******************************************************************************
//*******************************************************************************
