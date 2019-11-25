CXX = g++

CXXFLAGS += -c -g -Wall
LDFLAGS += $(shell pkg-config --libs --static opencv)

SRCS = main.cpp
OBJS = $(subst .cc,.o,$(SRCS))

OBJECTS = $(SRCS:.cpp=.o)
EXECUTABLE = ica

all: $(OBJECTS) $(EXECUTABLE)

$(EXECUTABLE) : $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS) -lstdc++

.cpp.o: *.h
	$(CC) $(CFLAGS) $< -o $@ $(CXXFLAGS)

clean :
	rm -f $(OBJECTS) $(EXECUTABLE)

.PHONY: all clean
