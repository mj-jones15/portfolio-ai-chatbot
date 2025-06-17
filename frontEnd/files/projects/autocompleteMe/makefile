HEADERS = autocomplete.h SortingList.h term.h powerString.h
SOURCES = $(HEADERS:.l=.cpp)
OBJECTS = $(HEADERS:.h=.o)
MAINSOURCE = Project2.cpp
MAINOBJ = $(MAINSOURCE:.cpp=.o)
EXEC = Project2
$(EXEC): $(OBJECTS) $(MAINOBJ)
	g++ $^ -o $@
%.o: %.cpp $(HEADERS)
	g++ -c $< -o $@
clean:
	rm -i *.o
	rm -i `find ./ -type f executable`
