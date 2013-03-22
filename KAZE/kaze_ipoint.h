
/**
 * @file Ipoint.h
 * @brief Class that defines a point of interest
 * @date Jan 21, 2012
 * @author Pablo F. Alcantarilla
 */

#ifndef _IPOINT_H_
#define _IPOINT_H_

//*************************************************************************************
//*************************************************************************************

// Includes
#include <vector>
#include <math.h>

// Ipoint Class Declaration
class Ipoint
{
	
public:
		
		// Coordinates of the detected interest point
		float xf,yf;	// Float coordinates
        int x,y;        // Integer coordinates
		
        // Detected scale of the interest point (sigma units)
		float scale;
	
        // Size of the image derivatives (pixel units)
		int sigma_size;

        // Feature detector response
		float dresponse;
		
		// Evolution time
		float tevolution;

		// Octave of the keypoint
		float octave;
		
		// Sublevel in each octave of the keypoint
		float sublevel;
		
		// Descriptor vector and size
		std::vector<float> descriptor;
		int descriptor_size;

		// Main orientation
		float angle;
		
		// Descriptor mode
		int descriptor_mode;
		
		// Sign of the laplacian (for faster matching)
		int laplacian;
		
		// Evolution Level
		unsigned int level;
		
		// Constructor
		Ipoint(void);
};

//*************************************************************************************
//*************************************************************************************

#endif

