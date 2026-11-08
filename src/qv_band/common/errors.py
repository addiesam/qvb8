"""Exception types shared by every QV-band format and both runtime paths.

These were defined inside pack/pack_qvb8.py, which made a format-specific
file the home of a repo-wide contract: run_inference.py -- the loader every
live QVB8 run goes through -- had to import UncertifiableTensor from the
QVB8 packer to catch it. Moving QVB8's container out to pack/_new_format/
would have dragged the production loader along with it.

Neither name mentions QVB8: Qvb8Error is what every format raises when a
tensor cannot be held to tau, and UncertifiableTensor is the narrower case
where no storage dtype has room for the certified bound. Defined ONCE here so
every module that raises or catches either shares the same class -- two
independent definitions of a same-named exception make `except` silently miss
instances raised by the other module's class.
"""

__all__ = ["Qvb8Error", "UncertifiableTensor"]


class Qvb8Error(ValueError):
    """Raised when a tensor cannot be represented under the requested tau."""


class UncertifiableTensor(Qvb8Error):
    """Raised when NO storage dtype available to a merged qv_band weight
    can hold the pack's certified error bound -- the source checkpoint's
    own precision is not enough to hold tau's remaining headroom after
    the grid decode. The caller must keep that tensor dense (unpacked)
    rather than ship a weight the certificate no longer covers.
    """
