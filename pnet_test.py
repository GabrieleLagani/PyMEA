import sys

# Load pythonnet
from clr_loader import find_runtimes
from pythonnet import load, get_runtime_info
load("coreclr")
import clr

# Load dlls
sys.path.append('./API/') # add api to path
dll_modules = []
dll_modules += ['3Brain.BioCamDriver', '3Brain.Common']
for m in dll_modules:
	clr.AddReference(m) # import dll into python
	
# Import C# Objects into Python from namespaces
import _3Brain
from _3Brain.BioCamDriver import BioCamPool, StimEndPoint, BioCamStimExternalEndPoint
from _3Brain.Common import ChCoord, RectangularStimPulse
import System
from System import Console
Console.WriteLine("Hello from C#")

# Print runtime information
print("Available Runtimes:")
print(list(find_runtimes()))
print("Runtime Directory:")
print(System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory())
print("Runtime Info:")
print(System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription)
print(get_runtime_info())

# Check APIs
print("Check BioCamPool.Activate():")
BioCamPool.Activate()
print("Ok!")
print("RectangularStimPulse.Default.Properties:")
print(RectangularStimPulse.Default.Properties)