CC=$(CXX) -std=c++11 -Igtest/include -I.
CFLAGS = -Wall -pedantic -O3 -Xpreprocessor -fopenmp
MFLAGS = -larmadillo -lomp

.PHONY : clean distclean

all: deepgen_ode.exe

%.o: %.cpp libscmpp.h
	$(CC) -c $(CFLAGS) $<

%.exe: %.o
	$(CC) -o $@ $(CFLAGS) $< $(MFLAGS)

# ============================================================
# PHONY targets

clean :
	rm -f *.o core gmon.out *.gcno

distclean : clean
	rm -f *.exe

