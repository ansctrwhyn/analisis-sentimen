$(document).ready(function() {
    $(window).scroll(function() {
        if ($(document).scrollTop() > 100) {
            $(".desktop-navigation").addClass("nav-top");
        } else {
            $(".desktop-navigation").removeClass("nav-top");
        }
    });

    $(".search-icon-btn").click(function() {
        $('#BlockSearchForm').addClass('popup-box-on');
    });

    $("#RemoveClass").click(function() {
        $('#BlockSearchForm').removeClass('popup-box-on');
    });
});

// show hide password
$(".toggle-password").click(function() {
    $(this).toggleClass("fa-eye fa-eye-slash");
    var input = $($(this).attr("toggle"));
    if (input.attr("type") == "password") {
        input.attr("type", "text");
    } else {
        input.attr("type", "password");
    }
});

// hide otomatis accordion collapse
// $(document).click(function (e) {
//     if (!$(e.target).is('.panel-body')) {
//         $('.collapse').collapse('hide');
//     }
// });

// browser style infut form
function uploadFile(target) {
    document.getElementById("file-name").innerHTML = target.files[0].name;
}

// show hide with select option
$(function() {
    $("#PermohonanSelector").change(function() {
        $(".permohonan-selected").hide();
        $("#" + $(this).val()).show();
    });
});

// Ajukan keberatan lagi
function AjukanKeberatan_Ya() {
    document.getElementById('DownloadDokumen_AjukanKeberatan').style.display = 'block';
}

function AjukanKeberatan_Tidak() {
    document.getElementById('DownloadDokumen_AjukanKeberatan').style.display = 'none';
}