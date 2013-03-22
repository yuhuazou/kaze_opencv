
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

#include "kaze_Ipoint.h"

//*******************************************************************************
//*******************************************************************************

/**
 * @brief Ipoint default constructor
 */
Ipoint::Ipoint(void)
{
	xf = yf = 0.0;
	x = y = 0;
	scale = 0.0;
	dresponse = 0.0;
	tevolution = 0.0;
	octave = 0.0;
	sublevel = 0.0;
	descriptor_size = 0;
	descriptor_mode = 0;
	laplacian = 0;
	level = 0;
}

//*******************************************************************************
//*******************************************************************************
