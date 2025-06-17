/*
 * Course: CS216
 * Project: Project 2
 * Purpose: IMDB Movie Name Searcher 
 */
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <stdexcept>
#include "term.h"
#include "SortingList.h"
#include "autocomplete.h"
#include "powerString.h"

using namespace std;

int main(int argc, char** argv)
{
	const int ARGUMENTS = 2;
	
	//check command line arguments
	if (argc != ARGUMENTS + 1)
	{
		cout << "Usage: " << argv[0] << " <filename> " << endl;
		return 1;
	}

	//check file opened correctly
	ifstream infile;
	infile.open(argv[1]);
	if(!infile.good())
	{
		cout << "Cannot open the file named " << argv[1] << endl;
		return 2;
	}

	//check number read correctly
	int numTerms;
	try 
	{
		numTerms = stoi(argv[2]); //Converts argument to int (from string) 
		if(numTerms < 1)
		{
			cout << "Please provide a positive number of matching terms!" << endl;
			return 3;
		}

	} catch (const invalid_argument& e) //Checks whether the argument correctly becomes an integer
	{
		cout << "Invalid input! Please provide an integer." << endl;
		return 3;
	} 


	//read in the terms from the input file
	//line by line and then store into autocomplete object
	Autocomplete autocomplete;
	long weight;
	string query;

	while(!infile.eof())
	{
		infile >> weight >> ws;
			getline(infile, query);
		if(query != "")
		{
			Term newterm(query, weight);
			autocomplete.insert(newterm);
		}
	}

	autocomplete.sort();
	//declare variables for user input
	string input;
	string prefix;
	//informative output
	cout << "Enjoy CS216 Auto-complete Me search engine!" << endl;
	cout << "It is written by Matthew Jones in CS216, Section 7?" << endl;
	cout << "Please input the search query (type \"exit\" to quit): " << endl;
	getline(cin, input); //read in user data
	powerString myPower(input); //create a powerString to clean the input
	prefix = myPower.wordFormat(); //assign cleaned input to prefix
	while(prefix != "Exit") //check if exitting program
	{
		//Filter List
		SortingList<Term> matchedTerms = autocomplete.allMatches(prefix);
		if(matchedTerms.size() == 0) //check if no terms matched
		{
			cout << "No matched query!" << endl;
		}
		else
		{
			if(numTerms > matchedTerms.size()) //check if user requested more terms than there were matches
			{
				cout << "Requested number of matches is greater than the number of found matches. Printing " << matchedTerms.size()
					 << " matches." << endl;
				for(int i = 0; i < matchedTerms.size(); i++)
				{
					cout << matchedTerms[i] << endl;
				}
			}
			else
			{
				//if not too many terms, simply output numTerms number of terms
				for(int i = 0; i < numTerms; i++)
				{
					cout << matchedTerms[i] << endl;
				}
			}	
		}
		cout << "Enjoy CS216 Auto-complete Me search engine!" << endl;
		cout << "It is written by Matthew Jones in CS216, Section 7?" << endl;
		cout << "Please input the search query (type \"exit\" to quit): " << endl;
		//take in new user input
		getline(cin, input);
		powerString myPower(input);
		prefix = myPower.wordFormat();
	}
	cout << "Thank you for using Auto-complete feature provided by Matthew Jones!" << endl;
	return 0;
}
