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

#include "particle_filter.h"

using namespace std;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Set the number of particles
	num_particles = 100;

	// Resize weights based on the number of particles
	weights.resize(num_particles, 1.0f);

	// Create a Gaussian/normal distribution for x, y, and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// Create the particles, set values based on the normal distribution, and set weights to 1.0
	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0f;
		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Create Gaussian/normal distributions for sensor noise
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);

	// Calculate the new state and add noise to each particle
	for (int i = 0; i < num_particles; i++) {
		// Compute the new state
		if (fabs(yaw_rate) < 0.00001) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		} else {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		// Add noise
		particles[i].x += noise_x(gen);
		particles[i].y += noise_y(gen);
		particles[i].theta += noise_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	int min_id;
	double d_x, d_y, min_distance;

	for (int i = 0; i < observations.size(); i++) {
		auto o = observations[i];
		min_id = -1;
		min_distance = INFINITY;

		for (int j = 0; j < predicted.size(); j++) {
			auto pred = predicted[i];

			// Get the distance between predicted and observed points
			d_x = pred.x - o.x;
			d_y = pred.y - o.y;
			distance = d_x * d_x + d_y * d_y;

			// Find the predicted landmark closest to the observed landmark
			if (distance < min_distance) {
				min_distance = distance;
				min_id = j;
			}
		}

		// Set the id of the observation to the id of the closest predicted landmark
		observations[i].id = min_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a multi-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// Iterate overl all particles
	for (int i = 0; i < num_particles; i++) {

		// Create a vector for locations that are within range of the particle
		vector<LandmarkObs> pred;

		// Get the coordinates
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		// Iterate over landmarks
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

			// Get landmark info
			int id = map_landmarks.landmark_list[j].id_i;
			float x = map_landmarks.landmark_list[j].x_f;
			float y = map_landmarks.landmark_list[j].y_f;

			// Get landmarks that are detected by the particle's sensor only
			if (fabs(x - p_x) <= sensor_range && fabs(y - p_y) <= sensor_range) {
				pred.push_back(LandmarkObs{id, x, y});
			}
		}

		// Copy the observations, transforming from vehicle to map coordinates
		vector<LandmarkObs> new_obs;
		for (int k = 0; k < observations.size(); k++) {
			int t_id = observations[k].id;
			double t_x = cos(p_theta) * observations[k].x - sin(p_theta) * observations[k].y + p_x;
			double t_y = sin(p_theta) * observations[k].x + cos(p_theta) * observations[k].y + p_y;
			new_obs.push_back(LandmarkObs{t_id, t_x, t_y});
		}

		// Run dataAssociation
		dataAssociation(pred, new_obs);

		// Set the weight back to 1.0
		particles[i].weight = 1.0f;

		for (int l = 0; l < new_obs.size(); l++) {
			double obs_x = new_obs[l].x;
			double obs_y = new_obs[l].y;
			double pred_x = 0;
			double pred_y = 0;
			int pred_id = new_obs[l].id;

			// Get the coordinates of the prediction for the current observation
			for (int m = 0; m < predictions.size(); k++) {
				if (pred_id == predictions[m].id) {
					pred_x = predictions[m].x;
					pred_y = predictions[m].y;
				}
			}

			// Get the weight for the current observation
			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
			particles[i].weight += 1/(2*M_PI*std_x*std_y) * exp( -1 * (pow(pred_x-obs_x,2)/(2*pow(std_x, 2)) + (pow(pred_y-obs_y,2)/(2*pow(sts_y, 2)))));
		}
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Vector for the new particles
	std::vector<Particle> updated_particles;

	// Compute the discrete distribution
	std::discrete_distribution<int> disc(weights.begin(), weights.end());

	// Assign the particles based on the discrete distribution
	for (int i = 0; i < num_particles; i++) {
		auto index = disc(gen);
		updated_particles.push_back(std::move(particles[index]))
	}

	// Set the particles vector to the updated particles
	particles = std::move(new_particles);
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
