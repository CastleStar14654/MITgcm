#include "AUTODIFF_OPTIONS.h"

CBOP
C     !ROUTINE: DUMMY_IN_STEPPING
C     !INTERFACE:
      SUBROUTINE DUMMY_IN_STEPPING( myTime, myIter, myThid )

C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE DUMMY_IN_STEPPING                             |
C     *==========================================================*
C     \ev

C     !USES:
      IMPLICIT NONE

C     == Global variables ===
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"

C     !INPUT/OUTPUT PARAMETERS:
C     myTime :: time counter for this thread
C     myIter :: iteration counter for this thread
C     myThid :: Thread number for this instance of the routine.
      _RL     myTime
      INTEGER myIter
      INTEGER myThid

C     !LOCAL VARIABLES:
CEOP

      RETURN
      END
