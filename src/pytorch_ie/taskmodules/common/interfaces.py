import abc
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

from pie_core import Annotation

# Annotation Encoding type: encoding for a single annotation
AE = TypeVar("AE")
# Annotation type
A = TypeVar("A", bound=Annotation)
# Annotation Collection Encoding type: encoding for a collection of annotations,
# e.g. all relevant annotations for a document
ACE = TypeVar("ACE")


class DecodingException(Exception, Generic[AE], abc.ABC):
    """Exception raised when decoding fails."""

    identifier: str

    def __init__(self, message: str, encoding: AE):
        self.message = message
        self.encoding = encoding


class AnnotationEncoderDecoder(abc.ABC, Generic[A, AE]):
    """Base class for annotation encoders and decoders."""

    @abc.abstractmethod
    def encode(self, annotation: A, metadata: Optional[Dict[str, Any]] = None) -> AE:
        pass

    @abc.abstractmethod
    def decode(self, encoding: AE, metadata: Optional[Dict[str, Any]] = None) -> A:
        pass

    def build_decoding_constraints(
        self, partial_encoding: AE
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Given a partial encoding, build the constraints for the next encoding step.

        Returns:
            - A tuple of two elements:
                - The first element is a set of positive constraints for the decoder.
                - The second element is a set of negative constraints for the decoder.
        """
        raise NotImplementedError(
            "build_decoder_constraints is not implemented for this encoder/decoder."
        )

    def parse(self, encoding: AE) -> Tuple[List[A], Dict[str, int], AE]:
        """Parse the encoding and return a list of annotations. This should be error tolerant and
        return all annotations that can be parsed and the remaining encoding.

        Args:
            encoding: The encoding to parse. Can be incomplete.

        Returns:
            - A tuple of three elements:
                - A list of encoded annotations.
                - A dictionary mapping error types to their counts.
                - The remaining encoding after parsing.
        """
        raise NotImplementedError("parse is not implemented for this encoder/decoder.")
