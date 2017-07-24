/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <cfloat>

#include "particle_filter.h"

#define MATH_PI 3.1415926
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	particles.resize(num_particles);
	weights.resize(num_particles);

	default_random_engine gen;

	// Creates a normal (Gaussian) distribution for x, y, and theta.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_t(theta, std[2]);

	for (int i = 0; i < num_particles; ++i)
	{
		Particle p;

		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_t(gen);
		p.weight = 1.0;

		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_t(0, std_pos[2]);
	
	double x1, y1;
	double delta_theta = delta_t * yaw_rate;
	for (int i = 0; i < num_particles; ++i)
	{
		Particle &p = particles[i];
		if(abs(yaw_rate) > 0.000001) {
			x1 = p.x + velocity * (sin(p.theta+delta_theta) - sin(p.theta)) / yaw_rate;
			y1 = p.y + velocity * (cos(p.theta) - cos(p.theta+delta_theta)) / yaw_rate;
		} else {
			x1 = p.x + velocity * cos(p.theta) * delta_t;
			y1 = p.y + velocity * sin(p.theta) * delta_t;
		}

		p.x = x1 + dist_x(gen);
		p.y = y1 + dist_y(gen);
		p.theta = p.theta + delta_theta + dist_t(gen);

		// cout << "Particle: x=" << p.x << ", y=" << p.y << "theta: " << p.theta << endl;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); ++i)
	{
		double min_distance = DBL_MAX;
		int nearest_id = 0;

		for (int k = 0; k < predicted.size(); ++k)
		{
			LandmarkObs mark = predicted[k];
			double distance = dist(mark.x, mark.y, observations[i].x, observations[i].y);
			if(distance < min_distance) {
				min_distance = distance;
				nearest_id = mark.id;
			}
		}

		observations[i].id = nearest_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// We will first need to transform the car's measurements from its local car coordinate system to the map's 
	// coordinate system. 
	// Next each measurement will need to be associated with a landmark identifier, for doing this part 
	// we will simply take the closest landmark to each transformed observation.

	if(!is_initialized) return;

	std::vector<LandmarkObs> transformedObs(observations.size());

	double std_x = std_landmark[0];
	double std_y = std_landmark[1];
	double total_weight = 0.0;
	for (int i = 0; i < particles.size(); ++i)
	{
		Particle p = particles[i];
		p.weight = 1.0;

		double final_weight = 1.0;
		for (int j = 0; j < observations.size(); ++j)
		{
			LandmarkObs obs = observations[j];
			LandmarkObs newObs;

			// Transform
			double x = p.x;
			double y = p.y;
			double theta = p.theta;
			newObs.x = obs.x * cos(theta) - obs.y * sin(theta) + x;
			newObs.y = obs.x * sin(theta) + obs.y * cos(theta) + y;
			transformedObs[j] = newObs; 

			// Find nearest landmark
			double min_distance = 1.0e30;
			int nearest_id = 0;
			double dx = 0.0;
			double dy = 0.0;
			for (int k = 0; k < map_landmarks.landmark_list.size(); ++k)
			{
				Map::single_landmark_s mark = map_landmarks.landmark_list[k];
				double distance = dist(mark.x_f, mark.y_f, newObs.x, newObs.y);
				if(distance < min_distance) {
					min_distance = distance;
					nearest_id = mark.id_i;
					dx = mark.x_f - newObs.x;
					dy = mark.y_f - newObs.y;
				}
			}
			newObs.id = nearest_id;

			//calculate individual measurement's probability
			//=(0.5/3.1415926/std_^2)*exp(-(0.5*(x1-x0)^2/std_^2+0.5*(y1-y0)^2/std_^2))
			double std_sq = std_x * std_y;
			double w = (0.5/MATH_PI/std_sq) * exp(-(dx*dx + dy*dy)/(2.0*std_sq));

			// The particles final weight will be calculated as the product of each 
			// measurement's Multivariate-Gaussian probability.
			final_weight *= w;
		}

		p.weight = final_weight;
		total_weight += final_weight; // sum up weights for all particles
		// cout << "weight: " << final_weight << endl;
		particles[i] = p;
	}
	// cout << "total_weight: " << total_weight << endl;

	// Normalize weights
	for (int i = 0; i < num_particles; ++i)
	{
		particles[i].weight /= total_weight;
		weights[i] = particles[i].weight;
		// cout << "normalized weight: " << particles[i].weight << endl;
	}
}

void ParticleFilter::resample() {
	if(!is_initialized) return;

	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<int> dist_w(weights.begin(), weights.end());

	std::vector<Particle> p2;
	for (int i=0; i<num_particles; ++i)
	{
		int next_index = dist_w(gen);
		Particle p = particles[next_index];
		p2.push_back(p);
	}
	particles = p2;
}


Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
