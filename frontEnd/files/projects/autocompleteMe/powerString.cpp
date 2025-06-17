/* 
 * File:   powerString.cpp
 * Course: CS216-00x
 * Project: Lab 8
 * Purpose: provide the implementation of member functions of powerString class
 */
#include <iostream> 
#include <algorithm>
#include "powerString.h"

using namespace std;

//constructor
powerString::powerString(string ini_string)
{
    this->str = ini_string;
}

// set the current value of str
void powerString::setPowerString(string value)
{
    this->str = value;
}

// return the value of str
string powerString::getPowerString() const
{
    return this->str;
}

//returns a string which transforms each character of str into lower case
//note that str has not been changed
//(algorithm header file is needed for the definition of transform function)
string powerString::toLower() const
{
    string result = this->str;
    transform(this->str.begin(), this->str.end(), result.begin(), ::tolower);
    return result;
}

//returns a string which transforms each character of str into upper case
// use recursive function to implement
//note that str has not been changed
string powerString::toUpper() const
{
    static int fcount = 0; // use a static variable to count the function calling
    fcount++;   // increase count by one while calling the function
    cout << "Calling the recursive function named " << __func__ << " : " << fcount  << " times." << endl;   // note that __func__ is macro for the funciton name, it may be different for different compiler
    // Base case:
    if (this->str.length() == 0)
        return this->str;
    // Recursive case:
    string first = this->str.substr(0,1);
    const char* front = first.c_str();
    char upperFront[LEN];
    string result;
    upperFront[0] = toupper(front[0]);
    upperFront[1] = '\0';
    result = upperFront;
    powerString part(this->str.substr(1, this->str.length()-1));
    return result + part.toUpper();
}

//it remove the extra blank spaces(including tab) from str
//It defines as:
//   1. it remains only one space if str contains more than
//      one continuous blank spaces between two non-space characters
//      For example, if str is "hello,     world",
//      it should change into "hello, world"
//   2. it removes all spaces before the first non-space character
//   3. it also removes all spaces after the last non-space character
void powerString::removeExtraSpace()
{
    // your code should start here...
    int i = 0;
    while(i < str.length() && (str[i] == ' ' || str[i] == '\t'))
    {
	    str.erase(i, 1);
    }
    for(int j = 0; j < str.length() - 1; j++)
    {
	    if(str[j] == ' ' && str[j+1] == ' ')
	    {
		    str.erase(j, 1);
		    j--;
	    }
    }
    int h = str.length() - 1;
    while(h >= 0 && (str[h] == ' ' || str[h] == '\t'))
    {
	    str.erase(h, 1);
	    h--;
    }
}

// return a string which transforms only the first letter of each word in str into upper case letter
// the word in str is defined as a sequence of characters starting with non-space character and ending right before the next space
// note that str has not been changed
string powerString::wordFormat() const
{
    // before the wordFormat transformation
    // make sure (1)the extra space(s) has been removed from str; 
    //           (2)apply to all lower case letters of str
    // your code should start here...
    string result = this->str;
    powerString myNewPower(result);
    myNewPower.removeExtraSpace(); 
    myNewPower.setPowerString(myNewPower.toLower());
    result = myNewPower.getPowerString();

    bool capNext = true;

    for(int i = 0; i < result.length(); i++)
    {
	    if(capNext && isalpha(result[i]))
	    {
		    result[i] = toupper(result[i]);
		    capNext = false;
	    }
	    else if(result[i] == ' ')
	    {
		    capNext = true;
	    }
    }

    return result;
}
